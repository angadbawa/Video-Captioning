import argparse
import logging
import sys
import os
from pathlib import Path
from typing import List, Optional
import numpy as np
import cv2
from tqdm import tqdm
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.config.config import Config
from src.models.encoder import create_feature_extractor


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )


class VideoFeatureExtractor:
    """Extract features from videos using pre-trained CNN models."""
    
    def __init__(
        self,
        model_type: str = "vgg16",
        device: Optional[torch.device] = None,
        batch_size: int = 32,
        frames_per_video: int = 80,
        img_size: tuple = (224, 224)
    ):
        """
        Initialize feature extractor.
        
        Args:
            model_type: Type of CNN model ('vgg16', 'resnet50')
            device: Device to run extraction on
            batch_size: Batch size for processing frames
            frames_per_video: Number of frames to extract per video
            img_size: Image size for CNN input
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.frames_per_video = frames_per_video
        self.img_size = img_size
        
        # Load pre-trained model
        self.model = self._load_model(model_type)
        self.model.to(self.device)
        self.model.eval()
        
        # Setup transforms
        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized {model_type} feature extractor on {self.device}")
    
    def _load_model(self, model_type: str) -> torch.nn.Module:
        """Load pre-trained CNN model."""
        if model_type.lower() == "vgg16":
            model = models.vgg16(pretrained=True)
            # Remove final classification layer
            model.classifier = torch.nn.Sequential(*list(model.classifier.children())[:-1])
        elif model_type.lower() == "resnet50":
            model = models.resnet50(pretrained=True)
            # Remove final classification layer
            model = torch.nn.Sequential(*list(model.children())[:-1])
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        return model
    
    def extract_frames(self, video_path: Path) -> List[np.ndarray]:
        """
        Extract frames from video.
        
        Args:
            video_path: Path to video file
            
        Returns:
            List of frame arrays
        """
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Calculate frame indices to sample uniformly
        if total_frames <= self.frames_per_video:
            frame_indices = list(range(total_frames))
        else:
            frame_indices = np.linspace(0, total_frames - 1, self.frames_per_video, dtype=int)
        
        frames = []
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            else:
                self.logger.warning(f"Failed to read frame {frame_idx} from {video_path}")
        
        cap.release()
        
        # Pad with last frame if needed
        while len(frames) < self.frames_per_video:
            if frames:
                frames.append(frames[-1].copy())
            else:
                # Create black frame if no frames were extracted
                frames.append(np.zeros((480, 640, 3), dtype=np.uint8))
        
        return frames[:self.frames_per_video]
    
    def extract_features_from_frames(self, frames: List[np.ndarray]) -> np.ndarray:
        """
        Extract CNN features from frames.
        
        Args:
            frames: List of frame arrays
            
        Returns:
            Feature array [num_frames, feature_dim]
        """
        # Convert frames to tensors
        frame_tensors = []
        for frame in frames:
            pil_frame = Image.fromarray(frame)
            tensor = self.transform(pil_frame)
            frame_tensors.append(tensor)
        
        # Stack into batch
        batch_tensor = torch.stack(frame_tensors).to(self.device)
        
        # Extract features in batches
        features = []
        with torch.no_grad():
            for i in range(0, len(batch_tensor), self.batch_size):
                batch = batch_tensor[i:i + self.batch_size]
                batch_features = self.model(batch)
                
                # Flatten features
                batch_features = batch_features.view(batch_features.size(0), -1)
                features.append(batch_features.cpu().numpy())
        
        return np.vstack(features)
    
    def extract_video_features(self, video_path: Path) -> np.ndarray:
        """
        Extract features from a single video.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Feature array [num_frames, feature_dim]
        """
        try:
            # Extract frames
            frames = self.extract_frames(video_path)
            
            # Extract features
            features = self.extract_features_from_frames(frames)
            
            return features
            
        except Exception as e:
            self.logger.error(f"Failed to extract features from {video_path}: {e}")
            raise
    
    def process_video_directory(
        self,
        video_dir: Path,
        output_dir: Path,
        video_extensions: List[str] = None
    ) -> List[Path]:
        """
        Process all videos in a directory.
        
        Args:
            video_dir: Directory containing videos
            output_dir: Directory to save features
            video_extensions: List of video file extensions to process
            
        Returns:
            List of processed feature file paths
        """
        if video_extensions is None:
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
        
        # Find all video files
        video_files = []
        for ext in video_extensions:
            video_files.extend(video_dir.glob(f'*{ext}'))
            video_files.extend(video_dir.glob(f'*{ext.upper()}'))
        
        self.logger.info(f"Found {len(video_files)} video files")
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process videos
        processed_files = []
        failed_files = []
        
        for video_path in tqdm(video_files, desc="Extracting features"):
            try:
                # Extract features
                features = self.extract_video_features(video_path)
                
                # Save features
                output_path = output_dir / f"{video_path.stem}.npy"
                np.save(output_path, features)
                processed_files.append(output_path)
                
            except Exception as e:
                self.logger.error(f"Failed to process {video_path}: {e}")
                failed_files.append(video_path)
        
        self.logger.info(f"Successfully processed {len(processed_files)} videos")
        if failed_files:
            self.logger.warning(f"Failed to process {len(failed_files)} videos")
        
        return processed_files


def create_dataset_csv(
    video_dir: Path,
    features_dir: Path,
    captions_file: Optional[Path],
    output_file: Path
):
    """
    Create dataset CSV file mapping videos to features and captions.
    
    Args:
        video_dir: Directory containing videos
        features_dir: Directory containing extracted features
        captions_file: Optional file containing captions
        output_file: Output CSV file path
    """
    logger = logging.getLogger(__name__)
    
    # Find all feature files
    feature_files = list(features_dir.glob('*.npy'))
    logger.info(f"Found {len(feature_files)} feature files")
    
    # Create dataset entries
    dataset_entries = []
    
    for feature_path in feature_files:
        video_id = feature_path.stem
        
        # Find corresponding video file
        video_path = None
        for ext in ['.mp4', '.avi', '.mov', '.mkv', '.wmv']:
            candidate_path = video_dir / f"{video_id}{ext}"
            if candidate_path.exists():
                video_path = candidate_path
                break
        
        if video_path is None:
            logger.warning(f"No video file found for feature file: {feature_path}")
            continue
        
        entry = {
            'video_id': video_id,
            'video_path': str(video_path),
            'feature_path': str(feature_path),
            'caption': ''  # Will be filled if captions file is provided
        }
        
        dataset_entries.append(entry)
    
    # Load captions if provided
    if captions_file and captions_file.exists():
        logger.info(f"Loading captions from {captions_file}")
        
        if captions_file.suffix == '.csv':
            captions_df = pd.read_csv(captions_file)
            
            # Try to match captions to videos
            for entry in dataset_entries:
                video_id = entry['video_id']
                
                # Look for matching caption
                matches = captions_df[captions_df['video_id'] == video_id]
                if not matches.empty:
                    entry['caption'] = matches.iloc[0]['caption']
                else:
                    # Try partial matching
                    partial_matches = captions_df[captions_df['video_id'].str.contains(video_id, na=False)]
                    if not partial_matches.empty:
                        entry['caption'] = partial_matches.iloc[0]['caption']
        
        elif captions_file.suffix == '.txt':
            # Assume one caption per line, matching order of feature files
            with open(captions_file, 'r') as f:
                captions = [line.strip() for line in f]
            
            for i, entry in enumerate(dataset_entries):
                if i < len(captions):
                    entry['caption'] = captions[i]
    
    # Create DataFrame and save
    df = pd.DataFrame(dataset_entries)
    df.to_csv(output_file, index=False)
    
    logger.info(f"Created dataset CSV with {len(df)} entries: {output_file}")
    
    # Print statistics
    with_captions = df[df['caption'] != ''].shape[0]
    logger.info(f"Entries with captions: {with_captions}/{len(df)}")


def main():
    """Main preprocessing function."""
    parser = argparse.ArgumentParser(description="Extract features from videos")
    parser.add_argument("--video-dir", type=str, required=True, help="Directory containing videos")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save features")
    parser.add_argument("--model-type", type=str, default="vgg16", choices=["vgg16", "resnet50"],
                       help="CNN model type for feature extraction")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for processing")
    parser.add_argument("--frames-per-video", type=int, default=80, help="Number of frames per video")
    parser.add_argument("--img-size", type=int, nargs=2, default=[224, 224], help="Image size for CNN")
    parser.add_argument("--device", type=str, help="Device to use (cuda/cpu)")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    
    # Dataset creation options
    parser.add_argument("--create-dataset", action="store_true", help="Create dataset CSV file")
    parser.add_argument("--captions-file", type=str, help="File containing captions")
    parser.add_argument("--dataset-output", type=str, default="dataset.csv", help="Output dataset CSV file")
    
    # Processing options
    parser.add_argument("--video-extensions", type=str, nargs="+", 
                       default=['.mp4', '.avi', '.mov', '.mkv', '.wmv'],
                       help="Video file extensions to process")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing feature files")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Setup paths
    video_dir = Path(args.video_dir)
    output_dir = Path(args.output_dir)
    
    if not video_dir.exists():
        raise ValueError(f"Video directory does not exist: {video_dir}")
    
    # Setup device
    device = torch.device(args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'))
    logger.info(f"Using device: {device}")
    
    try:
        # Initialize feature extractor
        extractor = VideoFeatureExtractor(
            model_type=args.model_type,
            device=device,
            batch_size=args.batch_size,
            frames_per_video=args.frames_per_video,
            img_size=tuple(args.img_size)
        )
        
        # Process videos
        logger.info("Starting feature extraction...")
        processed_files = extractor.process_video_directory(
            video_dir=video_dir,
            output_dir=output_dir,
            video_extensions=args.video_extensions
        )
        
        logger.info(f"Feature extraction completed. Processed {len(processed_files)} videos.")
        
        # Create dataset CSV if requested
        if args.create_dataset:
            logger.info("Creating dataset CSV...")
            captions_file = Path(args.captions_file) if args.captions_file else None
            
            create_dataset_csv(
                video_dir=video_dir,
                features_dir=output_dir,
                captions_file=captions_file,
                output_file=Path(args.dataset_output)
            )
        
        logger.info("Preprocessing completed successfully!")
        
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        raise


if __name__ == "__main__":
    main()
