"""Configuration management for Video Captioning project."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple
import torch


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    # Encoder configuration
    cnn_feature_dim: int = 4096
    encoder_hidden_dim: int = 512
    encoder_num_layers: int = 2
    encoder_dropout: float = 0.3
    
    # Decoder configuration
    decoder_hidden_dim: int = 512
    decoder_num_layers: int = 2
    decoder_dropout: float = 0.3
    vocab_size: int = 10000
    embedding_dim: int = 512
    
    # Attention configuration
    attention_dim: int = 512
    use_attention: bool = True
    
    # Sequence configuration
    max_sequence_length: int = 20
    video_sequence_length: int = 80


@dataclass
class DataConfig:
    """Data processing configuration."""
    # Paths
    data_root: Path = Path("data")
    video_dir: Path = Path("data/videos")
    features_dir: Path = Path("data/features")
    captions_file: Path = Path("data/captions.csv")
    
    # Video processing
    img_size: Tuple[int, int] = (224, 224)
    frames_per_video: int = 80
    frame_sampling_rate: int = 1
    
    # Data splits
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    
    # Vocabulary
    vocab_threshold: int = 5
    max_vocab_size: int = 10000
    
    # Special tokens
    pad_token: str = "<PAD>"
    start_token: str = "<START>"
    end_token: str = "<END>"
    unk_token: str = "<UNK>"


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Training parameters
    batch_size: int = 32
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    gradient_clip_norm: float = 5.0
    
    # Optimization
    optimizer: str = "adam"  # adam, adamw, sgd
    scheduler: str = "cosine"  # cosine, step, plateau
    warmup_epochs: int = 5
    
    # Loss configuration
    label_smoothing: float = 0.1
    
    # Validation and checkpointing
    val_every_n_epochs: int = 1
    save_every_n_epochs: int = 5
    early_stopping_patience: int = 10
    
    # Device configuration
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4
    pin_memory: bool = True


@dataclass
class InferenceConfig:
    """Inference configuration."""
    # Search strategies
    search_method: str = "beam"  # beam, greedy
    beam_size: int = 5
    max_length: int = 20
    length_penalty: float = 1.0
    
    # Output configuration
    remove_special_tokens: bool = True
    capitalize_first: bool = True


@dataclass
class ExperimentConfig:
    """Experiment tracking configuration."""
    # Experiment details
    experiment_name: str = "video_captioning"
    project_name: str = "video-captioning-pytorch"
    
    # Logging
    log_every_n_steps: int = 100
    use_wandb: bool = True
    use_tensorboard: bool = True
    
    # Checkpoints
    checkpoint_dir: Path = Path("checkpoints")
    best_model_path: Path = Path("checkpoints/best_model.pth")
    
    # Outputs
    output_dir: Path = Path("outputs")
    predictions_file: Path = Path("outputs/predictions.json")


@dataclass
class Config:
    """Main configuration class."""
    model: ModelConfig = ModelConfig()
    data: DataConfig = DataConfig()
    training: TrainingConfig = TrainingConfig()
    inference: InferenceConfig = InferenceConfig()
    experiment: ExperimentConfig = ExperimentConfig()
    
    def __post_init__(self):
        """Post-initialization validation and setup."""
        self.data.data_root.mkdir(exist_ok=True)
        self.data.video_dir.mkdir(exist_ok=True)
        self.data.features_dir.mkdir(exist_ok=True)
        self.experiment.checkpoint_dir.mkdir(exist_ok=True)
        self.experiment.output_dir.mkdir(exist_ok=True)
        
        # Validate splits
        total_split = self.data.train_split + self.data.val_split + self.data.test_split
        if abs(total_split - 1.0) > 1e-6:
            raise ValueError(f"Data splits must sum to 1.0, got {total_split}")
        
        self.model.vocab_size = self.data.max_vocab_size


def get_config() -> Config:
    """Get default configuration."""
    return Config()
