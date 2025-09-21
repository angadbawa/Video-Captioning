import os
import torch
import json
from pathlib import Path
from typing import Dict, Any, Optional
import logging


class CheckpointManager:
    """Manager for saving and loading model checkpoints."""
    
    def __init__(self, checkpoint_dir: Path):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
    
    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        epoch: int,
        metrics: Dict[str, float],
        is_best: bool = False,
        additional_info: Optional[Dict[str, Any]] = None
    ) -> Path:
        """
        Save model checkpoint.
        
        Args:
            model: Model to save
            optimizer: Optimizer state
            scheduler: Learning rate scheduler state
            epoch: Current epoch
            metrics: Validation metrics
            is_best: Whether this is the best model so far
            additional_info: Additional information to save
            
        Returns:
            Path to saved checkpoint
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'model_config': getattr(model, 'config', None)
        }
        
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        if additional_info:
            checkpoint.update(additional_info)
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch:04d}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best model at epoch {epoch}")
        
        # Save latest checkpoint
        latest_path = self.checkpoint_dir / "latest_checkpoint.pth"
        torch.save(checkpoint, latest_path)
        
        self.logger.info(f"Saved checkpoint at epoch {epoch}")
        
        # Clean up old checkpoints (keep last 5)
        self._cleanup_old_checkpoints()
        
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path: Path) -> Dict[str, Any]:
        """
        Load checkpoint from file.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Loaded checkpoint dictionary
        """
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.logger.info(f"Loaded checkpoint from {checkpoint_path}")
        
        return checkpoint
    
    def load_best_model(self) -> Optional[Dict[str, Any]]:
        """
        Load the best model checkpoint.
        
        Returns:
            Best model checkpoint or None if not found
        """
        best_path = self.checkpoint_dir / "best_model.pth"
        
        if best_path.exists():
            return self.load_checkpoint(best_path)
        else:
            self.logger.warning("Best model checkpoint not found")
            return None
    
    def load_latest_checkpoint(self) -> Optional[Dict[str, Any]]:
        """
        Load the latest checkpoint.
        
        Returns:
            Latest checkpoint or None if not found
        """
        latest_path = self.checkpoint_dir / "latest_checkpoint.pth"
        
        if latest_path.exists():
            return self.load_checkpoint(latest_path)
        else:
            self.logger.warning("Latest checkpoint not found")
            return None
    
    def list_checkpoints(self) -> list:
        """
        List all available checkpoints.
        
        Returns:
            List of checkpoint paths
        """
        checkpoint_pattern = "checkpoint_epoch_*.pth"
        checkpoints = list(self.checkpoint_dir.glob(checkpoint_pattern))
        checkpoints.sort()
        
        return checkpoints
    
    def _cleanup_old_checkpoints(self, keep_last: int = 5):
        """
        Clean up old checkpoints, keeping only the last N.
        
        Args:
            keep_last: Number of recent checkpoints to keep
        """
        checkpoints = self.list_checkpoints()
        
        if len(checkpoints) > keep_last:
            old_checkpoints = checkpoints[:-keep_last]
            
            for checkpoint_path in old_checkpoints:
                try:
                    checkpoint_path.unlink()
                    self.logger.debug(f"Removed old checkpoint: {checkpoint_path}")
                except OSError as e:
                    self.logger.warning(f"Failed to remove checkpoint {checkpoint_path}: {e}")
    
    def save_model_for_inference(
        self,
        model: torch.nn.Module,
        vocabulary: Any,
        config: Any,
        model_name: str = "model_for_inference.pth"
    ) -> Path:
        """
        Save model specifically for inference with all necessary components.
        
        Args:
            model: Trained model
            vocabulary: Vocabulary object
            config: Configuration object
            model_name: Name for the saved model file
            
        Returns:
            Path to saved inference model
        """
        inference_package = {
            'model_state_dict': model.state_dict(),
            'model_config': config,
            'vocabulary': {
                'word2idx': vocabulary.word2idx,
                'idx2word': vocabulary.idx2word,
                'special_tokens': {
                    'pad_token': vocabulary.pad_token,
                    'start_token': vocabulary.start_token,
                    'end_token': vocabulary.end_token,
                    'unk_token': vocabulary.unk_token,
                    'pad_idx': vocabulary.pad_idx,
                    'start_idx': vocabulary.start_idx,
                    'end_idx': vocabulary.end_idx,
                    'unk_idx': vocabulary.unk_idx
                }
            },
            'model_info': {
                'vocab_size': len(vocabulary),
                'trainable_parameters': model.get_trainable_parameters()
            }
        }
        
        inference_path = self.checkpoint_dir / model_name
        torch.save(inference_package, inference_path)
        
        self.logger.info(f"Saved inference model to {inference_path}")
        
        # Also save configuration as JSON for easy inspection
        config_path = self.checkpoint_dir / "model_config.json"
        try:
            config_dict = self._config_to_dict(config)
            with open(config_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
        except Exception as e:
            self.logger.warning(f"Failed to save config as JSON: {e}")
        
        return inference_path
    
    def load_model_for_inference(self, model_path: Path) -> Dict[str, Any]:
        """
        Load model package for inference.
        
        Args:
            model_path: Path to inference model file
            
        Returns:
            Dictionary containing model components
        """
        if not model_path.exists():
            raise FileNotFoundError(f"Inference model not found: {model_path}")
        
        inference_package = torch.load(model_path, map_location='cpu')
        self.logger.info(f"Loaded inference model from {model_path}")
        
        return inference_package
    
    def _config_to_dict(self, config: Any) -> Dict[str, Any]:
        """
        Convert config object to dictionary for JSON serialization.
        
        Args:
            config: Configuration object
            
        Returns:
            Dictionary representation of config
        """
        if hasattr(config, '__dict__'):
            config_dict = {}
            for key, value in config.__dict__.items():
                if hasattr(value, '__dict__'):
                    config_dict[key] = self._config_to_dict(value)
                elif isinstance(value, Path):
                    config_dict[key] = str(value)
                else:
                    try:
                        json.dumps(value)  # Test if serializable
                        config_dict[key] = value
                    except (TypeError, ValueError):
                        config_dict[key] = str(value)
            return config_dict
        else:
            return str(config)
    
    def get_checkpoint_info(self, checkpoint_path: Path) -> Dict[str, Any]:
        """
        Get information about a checkpoint without loading the full model.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Dictionary with checkpoint information
        """
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Load only the metadata
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        info = {
            'epoch': checkpoint.get('epoch', 'unknown'),
            'metrics': checkpoint.get('metrics', {}),
            'file_size': checkpoint_path.stat().st_size,
            'created_time': checkpoint_path.stat().st_mtime
        }
        
        # Add model parameter count if available
        if 'model_state_dict' in checkpoint:
            total_params = sum(
                p.numel() for p in checkpoint['model_state_dict'].values()
            )
            info['total_parameters'] = total_params
        
        return info
