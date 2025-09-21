import os
import time
import json
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import wandb

from ..config.config import Config
from ..models.video_captioning_model import VideoCaptioningModel
from ..data.vocabulary import Vocabulary
from ..utils.metrics import CaptionMetrics
from ..utils.checkpoint import CheckpointManager


class VideoCaptioningTrainer:
    """Trainer class for video captioning model."""
    
    def __init__(
        self,
        model: VideoCaptioningModel,
        config: Config,
        vocabulary: Vocabulary,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device
    ):
        """
        Initialize trainer.
        
        Args:
            model: Video captioning model
            config: Configuration object
            vocabulary: Vocabulary object
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Device to run training on
        """
        self.model = model.to(device)
        self.config = config
        self.vocabulary = vocabulary
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Initialize optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Initialize loss function
        self.criterion = self._create_loss_function()
        
        # Initialize metrics
        self.metrics = CaptionMetrics(vocabulary)
        
        # Initialize checkpoint manager
        self.checkpoint_manager = CheckpointManager(config.experiment.checkpoint_dir)
        
        # Initialize logging
        self.logger = self._setup_logging()
        self.tensorboard_writer = None
        self.use_wandb = config.experiment.use_wandb
        
        if config.experiment.use_tensorboard:
            self.tensorboard_writer = SummaryWriter(
                log_dir=config.experiment.checkpoint_dir / "tensorboard"
            )
        
        if self.use_wandb:
            wandb.init(
                project=config.experiment.project_name,
                name=config.experiment.experiment_name,
                config=self._get_wandb_config()
            )
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_score = 0.0
        self.patience_counter = 0
        
        # Training history
        self.train_history = []
        self.val_history = []
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer."""
        if self.config.training.optimizer.lower() == "adam":
            return optim.Adam(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay
            )
        elif self.config.training.optimizer.lower() == "adamw":
            return optim.AdamW(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay
            )
        elif self.config.training.optimizer.lower() == "sgd":
            return optim.SGD(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                momentum=0.9,
                weight_decay=self.config.training.weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.training.optimizer}")
    
    def _create_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler."""
        if self.config.training.scheduler.lower() == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.training.num_epochs,
                eta_min=self.config.training.learning_rate * 0.01
            )
        elif self.config.training.scheduler.lower() == "step":
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.training.num_epochs // 3,
                gamma=0.1
            )
        elif self.config.training.scheduler.lower() == "plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.5,
                patience=5,
                verbose=True
            )
        else:
            return None
    
    def _create_loss_function(self) -> nn.Module:
        """Create loss function."""
        return nn.CrossEntropyLoss(
            ignore_index=self.vocabulary.pad_idx,
            label_smoothing=self.config.training.label_smoothing
        )
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        # Create file handler
        log_file = self.config.experiment.checkpoint_dir / "training.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def _get_wandb_config(self) -> Dict[str, Any]:
        """Get configuration for wandb."""
        return {
            'model': {
                'encoder_hidden_dim': self.config.model.encoder_hidden_dim,
                'decoder_hidden_dim': self.config.model.decoder_hidden_dim,
                'embedding_dim': self.config.model.embedding_dim,
                'use_attention': self.config.model.use_attention,
                'vocab_size': self.config.model.vocab_size
            },
            'training': {
                'batch_size': self.config.training.batch_size,
                'learning_rate': self.config.training.learning_rate,
                'num_epochs': self.config.training.num_epochs,
                'optimizer': self.config.training.optimizer,
                'scheduler': self.config.training.scheduler
            },
            'data': {
                'max_sequence_length': self.config.model.max_sequence_length,
                'video_sequence_length': self.config.model.video_sequence_length
            }
        }
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(
                video_features=batch['video_features'],
                input_tokens=batch['input_tokens'],
                target_tokens=batch['target_tokens']
            )
            
            # Compute loss
            logits = outputs['logits']  # [batch_size, seq_len, vocab_size]
            targets = batch['target_tokens']  # [batch_size, seq_len]
            
            # Reshape for loss computation
            loss = self.criterion(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1)
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if self.config.training.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.gradient_clip_norm
                )
            
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            self.global_step += 1
            
            # Log progress
            if batch_idx % self.config.experiment.log_every_n_steps == 0:
                self.logger.info(
                    f"Epoch {self.current_epoch}, Batch {batch_idx}/{num_batches}, "
                    f"Loss: {loss.item():.4f}, LR: {self.optimizer.param_groups[0]['lr']:.6f}"
                )
                
                # Log to tensorboard
                if self.tensorboard_writer:
                    self.tensorboard_writer.add_scalar(
                        'Train/BatchLoss', loss.item(), self.global_step
                    )
                    self.tensorboard_writer.add_scalar(
                        'Train/LearningRate', 
                        self.optimizer.param_groups[0]['lr'], 
                        self.global_step
                    )
                
                # Log to wandb
                if self.use_wandb:
                    wandb.log({
                        'train/batch_loss': loss.item(),
                        'train/learning_rate': self.optimizer.param_groups[0]['lr'],
                        'global_step': self.global_step
                    })
        
        avg_loss = total_loss / num_batches
        return {'loss': avg_loss}
    
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_references = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(
                    video_features=batch['video_features'],
                    input_tokens=batch['input_tokens'],
                    target_tokens=batch['target_tokens']
                )
                
                # Compute loss
                logits = outputs['logits']
                targets = batch['target_tokens']
                
                loss = self.criterion(
                    logits.reshape(-1, logits.size(-1)),
                    targets.reshape(-1)
                )
                total_loss += loss.item()
                
                # Generate predictions for metrics
                generated_outputs = self.model.generate(
                    video_features=batch['video_features'],
                    start_token_id=self.vocabulary.start_idx,
                    end_token_id=self.vocabulary.end_idx,
                    max_length=self.config.model.max_sequence_length
                )
                
                # Decode predictions and references
                predictions = self._decode_sequences(
                    generated_outputs['generated_tokens']
                )
                references = self._decode_sequences(batch['target_tokens'])
                
                all_predictions.extend(predictions)
                all_references.extend(references)
        
        # Compute metrics
        avg_loss = total_loss / len(self.val_loader)
        metric_scores = self.metrics.compute_metrics(all_predictions, all_references)
        
        results = {'loss': avg_loss, **metric_scores}
        return results
    
    def _decode_sequences(self, sequences: torch.Tensor) -> list:
        """Decode token sequences to text."""
        decoded = []
        for seq in sequences:
            text = self.vocabulary.decode_caption(
                seq.cpu().tolist(), remove_special_tokens=True
            )
            decoded.append(text)
        return decoded
    
    def train(self) -> Dict[str, Any]:
        """Main training loop."""
        self.logger.info("Starting training...")
        self.logger.info(f"Model has {self.model.get_trainable_parameters():,} trainable parameters")
        
        start_time = time.time()
        
        for epoch in range(self.config.training.num_epochs):
            self.current_epoch = epoch
            
            # Train epoch
            train_metrics = self.train_epoch()
            
            # Validate epoch
            if epoch % self.config.training.val_every_n_epochs == 0:
                val_metrics = self.validate_epoch()
                
                # Log epoch results
                self.logger.info(
                    f"Epoch {epoch}: Train Loss: {train_metrics['loss']:.4f}, "
                    f"Val Loss: {val_metrics['loss']:.4f}, "
                    f"Val BLEU-4: {val_metrics.get('bleu_4', 0):.4f}"
                )
                
                # Log to tensorboard
                if self.tensorboard_writer:
                    self.tensorboard_writer.add_scalar(
                        'Train/EpochLoss', train_metrics['loss'], epoch
                    )
                    self.tensorboard_writer.add_scalar(
                        'Val/EpochLoss', val_metrics['loss'], epoch
                    )
                    for metric_name, metric_value in val_metrics.items():
                        if metric_name != 'loss':
                            self.tensorboard_writer.add_scalar(
                                f'Val/{metric_name}', metric_value, epoch
                            )
                
                # Log to wandb
                if self.use_wandb:
                    wandb.log({
                        'train/epoch_loss': train_metrics['loss'],
                        'val/epoch_loss': val_metrics['loss'],
                        **{f'val/{k}': v for k, v in val_metrics.items() if k != 'loss'},
                        'epoch': epoch
                    })
                
                # Check for improvement
                current_score = val_metrics.get('bleu_4', val_metrics['loss'])
                is_best = current_score > self.best_val_score
                
                if is_best:
                    self.best_val_score = current_score
                    self.patience_counter = 0
                    
                    # Save best model
                    self.checkpoint_manager.save_checkpoint(
                        model=self.model,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        epoch=epoch,
                        metrics=val_metrics,
                        is_best=True
                    )
                else:
                    self.patience_counter += 1
                
                # Early stopping
                if self.patience_counter >= self.config.training.early_stopping_patience:
                    self.logger.info(f"Early stopping at epoch {epoch}")
                    break
                
                # Save history
                self.train_history.append(train_metrics)
                self.val_history.append(val_metrics)
            
            # Update scheduler
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics.get('bleu_4', val_metrics['loss']))
                else:
                    self.scheduler.step()
            
            # Save regular checkpoint
            if epoch % self.config.training.save_every_n_epochs == 0:
                self.checkpoint_manager.save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch,
                    metrics=val_metrics if 'val_metrics' in locals() else {},
                    is_best=False
                )
        
        # Training completed
        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time:.2f} seconds")
        
        # Save final results
        results = {
            'best_val_score': self.best_val_score,
            'total_epochs': self.current_epoch + 1,
            'total_time': total_time,
            'train_history': self.train_history,
            'val_history': self.val_history
        }
        
        # Save results to file
        results_file = self.config.experiment.checkpoint_dir / "training_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Close logging
        if self.tensorboard_writer:
            self.tensorboard_writer.close()
        
        if self.use_wandb:
            wandb.finish()
        
        return results
    
    def load_checkpoint(self, checkpoint_path: Path) -> Dict[str, Any]:
        """Load checkpoint and resume training."""
        checkpoint = self.checkpoint_manager.load_checkpoint(checkpoint_path)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint.get('global_step', 0)
        self.best_val_score = checkpoint.get('best_val_score', 0.0)
        
        self.logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")
        
        return checkpoint
