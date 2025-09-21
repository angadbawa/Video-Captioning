import torch
import torch.nn.functional as F
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import logging
from PIL import Image
import torchvision.transforms as transforms

from ..config.config import Config
from ..models.video_captioning_model import VideoCaptioningModel
from ..data.vocabulary import Vocabulary
from ..utils.checkpoint import CheckpointManager


class VideoCaptionPredictor:
    """Predictor class for generating video captions."""
    
    def __init__(
        self,
        model_path: Path,
        device: Optional[torch.device] = None,
        config: Optional[Config] = None
    ):
        """
        Initialize predictor.
        
        Args:
            model_path: Path to trained model
            device: Device to run inference on
            config: Optional configuration override
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = logging.getLogger(__name__)
        
        # Load model and components
        self._load_model(model_path, config)
        
        # Setup image transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.logger.info(f"Predictor initialized on device: {self.device}")
    
    def _load_model(self, model_path: Path, config_override: Optional[Config] = None):
        """Load model and associated components."""
        checkpoint_manager = CheckpointManager(model_path.parent)
        
        # Load inference package
        package = checkpoint_manager.load_model_for_inference(model_path)
        
        # Extract components
        self.config = config_override or package['model_config']
        vocab_data = package['vocabulary']
        
        # Reconstruct vocabulary
        self.vocabulary = Vocabulary(self.config)
        self.vocabulary.word2idx = vocab_data['word2idx']
        self.vocabulary.idx2word = vocab_data['idx2word']
        
        # Update special token indices
        special_tokens = vocab_data['special_tokens']
        self.vocabulary.pad_idx = special_tokens['pad_idx']
        self.vocabulary.start_idx = special_tokens['start_idx']
        self.vocabulary.end_idx = special_tokens['end_idx']
        self.vocabulary.unk_idx = special_tokens['unk_idx']
        
        # Initialize and load model
        vocab_size = len(self.vocabulary)
        self.model = VideoCaptioningModel(self.config, vocab_size)
        self.model.load_state_dict(package['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        self.logger.info(f"Loaded model with {vocab_size} vocabulary size")
    
    def predict_from_features(
        self,
        video_features: np.ndarray,
        method: str = 'greedy',
        max_length: int = 20,
        beam_size: int = 5,
        length_penalty: float = 1.0,
        temperature: float = 1.0
    ) -> Dict[str, Union[str, List[str], torch.Tensor]]:
        """
        Generate caption from pre-extracted video features.
        
        Args:
            video_features: Video features array [seq_len, feature_dim]
            method: Generation method ('greedy', 'beam')
            max_length: Maximum caption length
            beam_size: Beam size for beam search
            length_penalty: Length penalty for beam search
            temperature: Temperature for sampling
            
        Returns:
            Dictionary containing generated caption and metadata
        """
        # Prepare features
        features_tensor = torch.FloatTensor(video_features).unsqueeze(0).to(self.device)
        
        # Ensure correct sequence length
        target_length = self.config.model.video_sequence_length
        if features_tensor.size(1) != target_length:
            features_tensor = self._resize_features(features_tensor, target_length)
        
        # Generate caption
        with torch.no_grad():
            if method == 'greedy':
                outputs = self.model.generate(
                    video_features=features_tensor,
                    start_token_id=self.vocabulary.start_idx,
                    end_token_id=self.vocabulary.end_idx,
                    max_length=max_length,
                    method='greedy',
                    temperature=temperature
                )
            elif method == 'beam':
                outputs = self.model.generate(
                    video_features=features_tensor,
                    start_token_id=self.vocabulary.start_idx,
                    end_token_id=self.vocabulary.end_idx,
                    max_length=max_length,
                    method='beam',
                    beam_size=beam_size,
                    length_penalty=length_penalty
                )
            else:
                raise ValueError(f"Unsupported generation method: {method}")
        
        # Decode caption
        generated_tokens = outputs['generated_tokens'][0].cpu().tolist()
        caption = self.vocabulary.decode_caption(generated_tokens, remove_special_tokens=True)
        
        result = {
            'caption': caption,
            'tokens': generated_tokens,
            'method': method
        }
        
        # Add attention weights if available
        if 'attention_weights' in outputs:
            result['attention_weights'] = outputs['attention_weights'][0].cpu()
        
        return result
    
    def predict_from_video(
        self,
        video_path: Path,
        method: str = 'greedy',
        max_length: int = 20,
        beam_size: int = 5,
        length_penalty: float = 1.0,
        temperature: float = 1.0,
        extract_features: bool = True
    ) -> Dict[str, Union[str, List[str], torch.Tensor]]:
        """
        Generate caption from video file.
        
        Args:
            video_path: Path to video file
            method: Generation method ('greedy', 'beam')
            max_length: Maximum caption length
            beam_size: Beam size for beam search
            length_penalty: Length penalty for beam search
            temperature: Temperature for sampling
            extract_features: Whether to extract features on-the-fly
            
        Returns:
            Dictionary containing generated caption and metadata
        """
        if extract_features:
            # Extract features from video
            video_features = self._extract_video_features(video_path)
        else:
            # Load pre-extracted features
            feature_path = video_path.with_suffix('.npy')
            if not feature_path.exists():
                raise FileNotFoundError(f"Feature file not found: {feature_path}")
            video_features = np.load(feature_path)
        
        # Generate caption
        result = self.predict_from_features(
            video_features=video_features,
            method=method,
            max_length=max_length,
            beam_size=beam_size,
            length_penalty=length_penalty,
            temperature=temperature
        )
        
        result['video_path'] = str(video_path)
        return result
    
    def predict_batch(
        self,
        video_features_list: List[np.ndarray],
        method: str = 'greedy',
        max_length: int = 20,
        beam_size: int = 5,
        length_penalty: float = 1.0,
        temperature: float = 1.0
    ) -> List[Dict[str, Union[str, List[str], torch.Tensor]]]:
        """
        Generate captions for a batch of videos.
        
        Args:
            video_features_list: List of video features arrays
            method: Generation method ('greedy', 'beam')
            max_length: Maximum caption length
            beam_size: Beam size for beam search
            length_penalty: Length penalty for beam search
            temperature: Temperature for sampling
            
        Returns:
            List of prediction results
        """
        results = []
        
        for video_features in video_features_list:
            result = self.predict_from_features(
                video_features=video_features,
                method=method,
                max_length=max_length,
                beam_size=beam_size,
                length_penalty=length_penalty,
                temperature=temperature
            )
            results.append(result)
        
        return results
    
    def _extract_video_features(self, video_path: Path) -> np.ndarray:
        """
        Extract features from video file.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Video features array [seq_len, feature_dim]
        """
        # This is a simplified feature extraction
        # In practice, you'd use a pre-trained CNN like VGG16 or ResNet
        frames = self._extract_frames(video_path)
        
        # Convert frames to features (placeholder - replace with actual CNN)
        features = []
        for frame in frames:
            # Flatten frame as a simple feature (replace with CNN features)
            feature = frame.flatten()[:4096]  # Truncate to 4096 dimensions
            if len(feature) < 4096:
                # Pad if necessary
                feature = np.pad(feature, (0, 4096 - len(feature)))
            features.append(feature)
        
        return np.array(features)
    
    def _extract_frames(self, video_path: Path) -> List[np.ndarray]:
        """
        Extract frames from video.
        
        Args:
            video_path: Path to video file
            
        Returns:
            List of frame arrays
        """
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        
        # Get total frame count
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        target_frames = self.config.model.video_sequence_length
        
        # Calculate frame indices to sample
        if total_frames > target_frames:
            indices = np.linspace(0, total_frames - 1, target_frames, dtype=int)
        else:
            indices = list(range(total_frames))
        
        # Extract frames
        for frame_idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (224, 224))
                frames.append(frame)
        
        cap.release()
        
        # Pad with zeros if needed
        while len(frames) < target_frames:
            frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
        
        return frames
    
    def _resize_features(self, features: torch.Tensor, target_length: int) -> torch.Tensor:
        """
        Resize feature sequence to target length.
        
        Args:
            features: Feature tensor [batch_size, seq_len, feature_dim]
            target_length: Target sequence length
            
        Returns:
            Resized feature tensor
        """
        batch_size, seq_len, feature_dim = features.shape
        
        if seq_len == target_length:
            return features
        elif seq_len > target_length:
            # Sample uniformly
            indices = torch.linspace(0, seq_len - 1, target_length, dtype=torch.long)
            return features[:, indices, :]
        else:
            # Pad with zeros
            padding = torch.zeros(batch_size, target_length - seq_len, feature_dim, 
                                device=features.device)
            return torch.cat([features, padding], dim=1)
    
    def generate_multiple_captions(
        self,
        video_features: np.ndarray,
        num_captions: int = 5,
        method: str = 'beam',
        max_length: int = 20,
        beam_size: int = 10,
        temperature: float = 1.0
    ) -> List[Dict[str, Union[str, float]]]:
        """
        Generate multiple diverse captions for a single video.
        
        Args:
            video_features: Video features array
            num_captions: Number of captions to generate
            method: Generation method
            max_length: Maximum caption length
            beam_size: Beam size (should be >= num_captions)
            temperature: Temperature for sampling
            
        Returns:
            List of caption dictionaries with scores
        """
        if method == 'beam' and beam_size < num_captions:
            beam_size = num_captions
        
        captions = []
        
        if method == 'beam':
            # Generate with beam search and return top-k
            result = self.predict_from_features(
                video_features=video_features,
                method='beam',
                max_length=max_length,
                beam_size=beam_size
            )
            
            # For now, return the single best caption
            # In a full implementation, you'd modify beam search to return multiple hypotheses
            captions.append({
                'caption': result['caption'],
                'score': 1.0,
                'tokens': result['tokens']
            })
        
        else:
            # Generate multiple captions with different temperatures
            temperatures = np.linspace(0.7, 1.3, num_captions)
            
            for temp in temperatures:
                result = self.predict_from_features(
                    video_features=video_features,
                    method='greedy',
                    max_length=max_length,
                    temperature=temp
                )
                
                captions.append({
                    'caption': result['caption'],
                    'score': 1.0 / temp,  # Higher score for lower temperature
                    'tokens': result['tokens'],
                    'temperature': temp
                })
        
        return captions
    
    def explain_prediction(
        self,
        video_features: np.ndarray,
        caption_tokens: List[int]
    ) -> Dict[str, torch.Tensor]:
        """
        Generate attention visualizations for a prediction.
        
        Args:
            video_features: Video features array
            caption_tokens: Generated caption tokens
            
        Returns:
            Dictionary containing attention weights and metadata
        """
        features_tensor = torch.FloatTensor(video_features).unsqueeze(0).to(self.device)
        target_length = self.config.model.video_sequence_length
        
        if features_tensor.size(1) != target_length:
            features_tensor = self._resize_features(features_tensor, target_length)
        
        # Forward pass to get attention weights
        with torch.no_grad():
            # Convert tokens to tensor
            input_tokens = torch.LongTensor(caption_tokens[:-1]).unsqueeze(0).to(self.device)
            target_tokens = torch.LongTensor(caption_tokens[1:]).unsqueeze(0).to(self.device)
            
            outputs = self.model(
                video_features=features_tensor,
                input_tokens=input_tokens,
                target_tokens=target_tokens
            )
        
        result = {
            'attention_weights': outputs.get('attention_weights'),
            'encoder_outputs': outputs.get('encoder_outputs'),
            'video_length': features_tensor.size(1),
            'caption_length': len(caption_tokens)
        }
        
        return result


class BatchPredictor:
    """Batch predictor for processing multiple videos efficiently."""
    
    def __init__(self, predictor: VideoCaptionPredictor, batch_size: int = 8):
        """
        Initialize batch predictor.
        
        Args:
            predictor: Single video predictor
            batch_size: Batch size for processing
        """
        self.predictor = predictor
        self.batch_size = batch_size
        self.logger = logging.getLogger(__name__)
    
    def predict_videos(
        self,
        video_paths: List[Path],
        method: str = 'greedy',
        max_length: int = 20,
        **kwargs
    ) -> List[Dict[str, Union[str, List[str]]]]:
        """
        Predict captions for multiple videos.
        
        Args:
            video_paths: List of video file paths
            method: Generation method
            max_length: Maximum caption length
            **kwargs: Additional arguments for prediction
            
        Returns:
            List of prediction results
        """
        results = []
        
        for i in range(0, len(video_paths), self.batch_size):
            batch_paths = video_paths[i:i + self.batch_size]
            
            self.logger.info(f"Processing batch {i//self.batch_size + 1}/{(len(video_paths) + self.batch_size - 1)//self.batch_size}")
            
            batch_results = []
            for video_path in batch_paths:
                try:
                    result = self.predictor.predict_from_video(
                        video_path=video_path,
                        method=method,
                        max_length=max_length,
                        **kwargs
                    )
                    batch_results.append(result)
                except Exception as e:
                    self.logger.error(f"Error processing {video_path}: {e}")
                    batch_results.append({
                        'video_path': str(video_path),
                        'caption': '',
                        'error': str(e)
                    })
            
            results.extend(batch_results)
        
        return results
