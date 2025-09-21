import argparse
import logging
import sys
from pathlib import Path
import torch
import pandas as pd
from sklearn.model_selection import train_test_split

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.config.config import Config
from src.data.vocabulary import Vocabulary, build_vocabulary_from_csv
from src.data.dataset import create_data_loaders
from src.models.video_captioning_model import VideoCaptioningModel
from src.training.trainer import VideoCaptioningTrainer
from src.utils.checkpoint import CheckpointManager


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('training.log')
        ]
    )


def prepare_data(config: Config) -> tuple:
    """Prepare training data."""
    logger = logging.getLogger(__name__)
    
    # Load data
    logger.info(f"Loading data from {config.data.captions_file}")
    df = pd.read_csv(config.data.captions_file)
    
    # Ensure required columns exist
    required_columns = ['video_id', 'caption', 'feature_path']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Filter out missing feature files
    df = df[df['feature_path'].apply(lambda x: Path(x).exists())]
    logger.info(f"Found {len(df)} samples with valid feature files")
    
    # Split data
    train_df, temp_df = train_test_split(
        df, test_size=(config.data.val_split + config.data.test_split),
        random_state=42, stratify=None
    )
    
    val_test_split = config.data.val_split / (config.data.val_split + config.data.test_split)
    val_df, test_df = train_test_split(
        temp_df, test_size=(1 - val_test_split),
        random_state=42, stratify=None
    )
    
    logger.info(f"Data splits - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    return train_df, val_df, test_df


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train video captioning model")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--data-file", type=str, required=True, help="Path to captions CSV file")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    parser.add_argument("--batch-size", type=int, help="Batch size override")
    parser.add_argument("--learning-rate", type=float, help="Learning rate override")
    parser.add_argument("--epochs", type=int, help="Number of epochs override")
    parser.add_argument("--device", type=str, help="Device to use (cuda/cpu)")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    parser.add_argument("--wandb", action="store_true", help="Use Weights & Biases logging")
    parser.add_argument("--no-tensorboard", action="store_true", help="Disable TensorBoard logging")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Load configuration
    config = Config()
    
    # Override config with command line arguments
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.learning_rate:
        config.training.learning_rate = args.learning_rate
    if args.epochs:
        config.training.num_epochs = args.epochs
    if args.device:
        config.training.device = args.device
    if args.wandb:
        config.experiment.use_wandb = True
    if args.no_tensorboard:
        config.experiment.use_tensorboard = False
    
    # Update paths
    config.data.captions_file = Path(args.data_file)
    config.experiment.checkpoint_dir = Path(args.checkpoint_dir)
    
    # Setup device
    device = torch.device(config.training.device)
    logger.info(f"Using device: {device}")
    
    # Prepare data
    logger.info("Preparing data...")
    train_df, val_df, test_df = prepare_data(config)
    
    # Build vocabulary
    logger.info("Building vocabulary...")
    vocabulary_path = config.experiment.checkpoint_dir / "vocabulary.json"
    
    if vocabulary_path.exists():
        logger.info("Loading existing vocabulary...")
        vocabulary = Vocabulary.load(vocabulary_path, config)
    else:
        logger.info("Building new vocabulary...")
        vocabulary = build_vocabulary_from_csv(config.data.captions_file, config, 'caption')
        vocabulary.save(vocabulary_path)
    
    # Update config with actual vocabulary size
    config.model.vocab_size = len(vocabulary)
    
    # Create data loaders
    logger.info("Creating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(
        config, vocabulary, train_df, val_df, test_df
    )
    
    # Initialize model
    logger.info("Initializing model...")
    model = VideoCaptioningModel(config, len(vocabulary))
    logger.info(f"Model has {model.get_trainable_parameters():,} trainable parameters")
    
    # Initialize trainer
    trainer = VideoCaptioningTrainer(
        model=model,
        config=config,
        vocabulary=vocabulary,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(Path(args.resume))
    
    # Start training
    logger.info("Starting training...")
    try:
        results = trainer.train()
        logger.info("Training completed successfully!")
        logger.info(f"Best validation score: {results['best_val_score']:.4f}")
        
        # Save final model for inference
        checkpoint_manager = CheckpointManager(config.experiment.checkpoint_dir)
        inference_model_path = checkpoint_manager.save_model_for_inference(
            model=model,
            vocabulary=vocabulary,
            config=config
        )
        logger.info(f"Saved inference model to: {inference_model_path}")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        # Save current state
        checkpoint_manager = CheckpointManager(config.experiment.checkpoint_dir)
        checkpoint_manager.save_checkpoint(
            model=model,
            optimizer=trainer.optimizer,
            scheduler=trainer.scheduler,
            epoch=trainer.current_epoch,
            metrics={},
            is_best=False
        )
        logger.info("Saved current training state")
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
