import os
import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import cv2
from PIL import Image
import torchvision.transforms as transforms

from ..config.config import Config
from .vocabulary import Vocabulary


class VideoCaptioningDataset(Dataset):
    """Dataset class for video captioning."""
    
    def __init__(
        self,
        data_df: pd.DataFrame,
        vocabulary: Vocabulary,
        config: Config,
        split: str = "train",
        transform: Optional[transforms.Compose] = None
    ):
        """
        Initialize the dataset.
        
        Args:
            data_df: DataFrame with video paths and captions
            vocabulary: Vocabulary object for text processing
            config: Configuration object
            split: Dataset split ('train', 'val', 'test')
            transform: Optional image transforms
        """
        self.data_df = data_df.reset_index(drop=True)
        self.vocabulary = vocabulary
        self.config = config
        self.split = split
        
        # Set up transforms
        if transform is None:
            self.transform = self._get_default_transforms()
        else:
            self.transform = transform
            
        # Validate data
        self._validate_data()
    
    def _get_default_transforms(self) -> transforms.Compose:
        """Get default image transforms."""
        if self.split == "train":
            return transforms.Compose([
                transforms.Resize(self.config.data.img_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.Resize(self.config.data.img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
    
    def _validate_data(self):
        """Validate dataset integrity."""
        missing_features = []
        for idx, row in self.data_df.iterrows():
            feature_path = row['feature_path']
            if not os.path.exists(feature_path):
                missing_features.append(feature_path)
        
        if missing_features:
            print(f"Warning: {len(missing_features)} feature files not found")
            # Remove missing files
            self.data_df = self.data_df[
                self.data_df['feature_path'].apply(os.path.exists)
            ].reset_index(drop=True)
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.data_df)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single item from the dataset.
        
        Args:
            idx: Index of the item
            
        Returns:
            Dictionary containing video features, caption tokens, and metadata
        """
        row = self.data_df.iloc[idx]
        
        # Load video features
        video_features = self._load_video_features(row['feature_path'])
        
        # Process caption
        caption = row['caption']
        caption_tokens = self.vocabulary.encode_caption(caption)
        
        # Create input and target sequences
        input_tokens = caption_tokens[:-1]  # Remove last token for input
        target_tokens = caption_tokens[1:]  # Remove first token for target
        
        # Pad sequences
        input_tokens = self._pad_sequence(input_tokens, self.config.model.max_sequence_length)
        target_tokens = self._pad_sequence(target_tokens, self.config.model.max_sequence_length)
        
        # Create attention mask for caption
        caption_mask = (input_tokens != self.vocabulary.pad_idx).float()
        
        return {
            'video_features': torch.FloatTensor(video_features),
            'input_tokens': torch.LongTensor(input_tokens),
            'target_tokens': torch.LongTensor(target_tokens),
            'caption_mask': caption_mask,
            'video_id': row.get('video_id', f'video_{idx}'),
            'caption_text': caption
        }
    
    def _load_video_features(self, feature_path: str) -> np.ndarray:
        """
        Load pre-extracted video features.
        
        Args:
            feature_path: Path to the feature file
            
        Returns:
            Video features array
        """
        features = np.load(feature_path)
        
        # Ensure consistent sequence length
        if len(features) > self.config.data.frames_per_video:
            # Sample frames uniformly
            indices = np.linspace(0, len(features) - 1, 
                                self.config.data.frames_per_video, dtype=int)
            features = features[indices]
        elif len(features) < self.config.data.frames_per_video:
            # Pad with zeros
            padding = np.zeros((
                self.config.data.frames_per_video - len(features),
                features.shape[1]
            ))
            features = np.vstack([features, padding])
        
        return features
    
    def _pad_sequence(self, sequence: List[int], max_length: int) -> List[int]:
        """
        Pad sequence to maximum length.
        
        Args:
            sequence: Input sequence
            max_length: Maximum sequence length
            
        Returns:
            Padded sequence
        """
        if len(sequence) > max_length:
            return sequence[:max_length]
        else:
            return sequence + [self.vocabulary.pad_idx] * (max_length - len(sequence))


class VideoFeatureDataset(Dataset):
    """Dataset for loading raw videos and extracting features on-the-fly."""
    
    def __init__(
        self,
        video_paths: List[str],
        config: Config,
        transform: Optional[transforms.Compose] = None
    ):
        """
        Initialize the dataset.
        
        Args:
            video_paths: List of video file paths
            config: Configuration object
            transform: Optional image transforms
        """
        self.video_paths = video_paths
        self.config = config
        
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(config.data.img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.video_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        """
        Get a single video and extract frames.
        
        Args:
            idx: Index of the video
            
        Returns:
            Dictionary containing video frames and metadata
        """
        video_path = self.video_paths[idx]
        frames = self._extract_frames(video_path)
        
        return {
            'frames': torch.stack(frames),
            'video_path': video_path,
            'video_id': Path(video_path).stem
        }
    
    def _extract_frames(self, video_path: str) -> List[torch.Tensor]:
        """
        Extract frames from video.
        
        Args:
            video_path: Path to video file
            
        Returns:
            List of frame tensors
        """
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        # Get total frame count
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame indices to sample
        if total_frames > self.config.data.frames_per_video:
            indices = np.linspace(0, total_frames - 1, 
                                self.config.data.frames_per_video, dtype=int)
        else:
            indices = list(range(total_frames))
        
        # Extract frames
        for i, frame_idx in enumerate(indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
                
                # Apply transforms
                frame_tensor = self.transform(frame)
                frames.append(frame_tensor)
        
        cap.release()
        
        # Pad with zeros if needed
        while len(frames) < self.config.data.frames_per_video:
            frames.append(torch.zeros_like(frames[0]))
        
        return frames


def create_data_loaders(
    config: Config,
    vocabulary: Vocabulary,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame] = None
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """
    Create data loaders for training, validation, and testing.
    
    Args:
        config: Configuration object
        vocabulary: Vocabulary object
        train_df: Training data DataFrame
        val_df: Validation data DataFrame
        test_df: Optional test data DataFrame
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create datasets
    train_dataset = VideoCaptioningDataset(
        train_df, vocabulary, config, split="train"
    )
    val_dataset = VideoCaptioningDataset(
        val_df, vocabulary, config, split="val"
    )
    
    test_dataset = None
    if test_df is not None:
        test_dataset = VideoCaptioningDataset(
            test_df, vocabulary, config, split="test"
        )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory,
        drop_last=False
    )
    
    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.training.batch_size,
            shuffle=False,
            num_workers=config.training.num_workers,
            pin_memory=config.training.pin_memory,
            drop_last=False
        )
    
    return train_loader, val_loader, test_loader
