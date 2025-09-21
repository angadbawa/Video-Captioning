import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import torchvision.models as models

from ..config.config import Config


class VideoEncoder(nn.Module):
    """Video encoder using LSTM to process video features."""
    
    def __init__(self, config: Config):
        """
        Initialize video encoder.
        
        Args:
            config: Configuration object
        """
        super(VideoEncoder, self).__init__()
        
        self.config = config
        self.feature_dim = config.model.cnn_feature_dim
        self.hidden_dim = config.model.encoder_hidden_dim
        self.num_layers = config.model.encoder_num_layers
        self.dropout = config.model.encoder_dropout
        
        # Feature projection layer
        self.feature_projection = nn.Linear(
            self.feature_dim, 
            self.hidden_dim
        )
        
        # LSTM encoder
        self.lstm = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Output projection (bidirectional LSTM doubles hidden size)
        self.output_projection = nn.Linear(
            self.hidden_dim * 2,  # Bidirectional
            self.hidden_dim
        )
        
        self.dropout_layer = nn.Dropout(self.dropout)
        
    def forward(
        self, 
        video_features: torch.Tensor,
        video_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of video encoder.
        
        Args:
            video_features: Video features [batch_size, seq_len, feature_dim]
            video_mask: Optional mask for video sequences [batch_size, seq_len]
            
        Returns:
            Tuple of (encoded_features, final_hidden_state)
        """
        batch_size, seq_len, _ = video_features.shape
        
        # Project features to hidden dimension
        projected_features = self.feature_projection(video_features)
        projected_features = self.dropout_layer(projected_features)
        
        # Pack sequences if mask is provided
        if video_mask is not None:
            lengths = video_mask.sum(dim=1).cpu()
            packed_features = nn.utils.rnn.pack_padded_sequence(
                projected_features, lengths, batch_first=True, enforce_sorted=False
            )
            packed_output, (hidden, cell) = self.lstm(packed_features)
            lstm_output, _ = nn.utils.rnn.pad_packed_sequence(
                packed_output, batch_first=True
            )
        else:
            lstm_output, (hidden, cell) = self.lstm(projected_features)
        
        # Project output to desired dimension
        encoded_features = self.output_projection(lstm_output)
        encoded_features = self.dropout_layer(encoded_features)
        
        # Combine forward and backward hidden states
        # hidden: [num_layers * 2, batch_size, hidden_dim]
        final_hidden = hidden[-2:].transpose(0, 1).contiguous().view(
            batch_size, -1
        )  # [batch_size, hidden_dim * 2]
        
        final_hidden = self.output_projection(final_hidden)
        
        return encoded_features, final_hidden


class CNNFeatureExtractor(nn.Module):
    """CNN feature extractor for videos."""
    
    def __init__(self, config: Config, pretrained: bool = True):
        """
        Initialize CNN feature extractor.
        
        Args:
            config: Configuration object
            pretrained: Whether to use pretrained weights
        """
        super(CNNFeatureExtractor, self).__init__()
        
        self.config = config
        
        # Load pretrained VGG16
        vgg16 = models.vgg16(pretrained=pretrained)
        
        # Remove the final classification layer
        self.features = vgg16.features
        self.avgpool = vgg16.avgpool
        self.classifier = nn.Sequential(*list(vgg16.classifier.children())[:-1])
        
        # Freeze feature extraction layers if using pretrained weights
        if pretrained:
            for param in self.features.parameters():
                param.requires_grad = False
            for param in self.avgpool.parameters():
                param.requires_grad = False
            for param in self.classifier.parameters():
                param.requires_grad = False
    
    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Extract features from video frames.
        
        Args:
            frames: Video frames [batch_size, num_frames, channels, height, width]
            
        Returns:
            Extracted features [batch_size, num_frames, feature_dim]
        """
        batch_size, num_frames, channels, height, width = frames.shape
        
        # Reshape to process all frames at once
        frames = frames.view(batch_size * num_frames, channels, height, width)
        
        # Extract features
        x = self.features(frames)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        
        # Reshape back to video format
        features = x.view(batch_size, num_frames, -1)
        
        return features


class ResNetFeatureExtractor(nn.Module):
    """ResNet feature extractor for videos."""
    
    def __init__(self, config: Config, pretrained: bool = True):
        """
        Initialize ResNet feature extractor.
        
        Args:
            config: Configuration object
            pretrained: Whether to use pretrained weights
        """
        super(ResNetFeatureExtractor, self).__init__()
        
        self.config = config
        
        # Load pretrained ResNet50
        resnet = models.resnet50(pretrained=pretrained)
        
        # Remove the final classification layer
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        
        # Freeze feature extraction layers if using pretrained weights
        if pretrained:
            for param in self.features.parameters():
                param.requires_grad = False
    
    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Extract features from video frames.
        
        Args:
            frames: Video frames [batch_size, num_frames, channels, height, width]
            
        Returns:
            Extracted features [batch_size, num_frames, feature_dim]
        """
        batch_size, num_frames, channels, height, width = frames.shape
        
        # Reshape to process all frames at once
        frames = frames.view(batch_size * num_frames, channels, height, width)
        
        # Extract features
        features = self.features(frames)
        features = features.view(batch_size, num_frames, -1)
        
        return features


def create_feature_extractor(
    config: Config, 
    model_type: str = "vgg16", 
    pretrained: bool = True
) -> nn.Module:
    """
    Create a feature extractor model.
    
    Args:
        config: Configuration object
        model_type: Type of model ('vgg16', 'resnet50')
        pretrained: Whether to use pretrained weights
        
    Returns:
        Feature extractor model
    """
    if model_type.lower() == "vgg16":
        return CNNFeatureExtractor(config, pretrained)
    elif model_type.lower() == "resnet50":
        return ResNetFeatureExtractor(config, pretrained)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
