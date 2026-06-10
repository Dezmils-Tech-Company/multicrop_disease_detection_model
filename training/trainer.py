"""Main training loop."""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Tuple, Optional
import csv
import json
from datetime import datetime

from .loss_functions import get_loss_function
from .metrics import MetricsCalculator
from .callbacks import EarlyStopping, ModelCheckpoint, LRScheduler


class Trainer:
    """Main training orchestrator."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 50,
        learning_rate: float = 1e-4,
        optimizer_type: str = "adam",
        loss_type: str = "cross_entropy",
        class_weights: Optional[torch.Tensor] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        checkpoint_dir: str = "checkpoints",
        log_dir: str = "logs",
        early_stopping_patience: int = 10,
        weight_decay: float = 0.0,
        scheduler_type: str = "cosine",
        scheduler_kwargs: Optional[dict] = None,
        checkpoint_monitor: str = "val_accuracy",
        save_best_only: bool = True,
    ):
        """
        Args:
            model: PyTorch model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            optimizer_type: 'adam' or 'sgd'
            loss_type: 'cross_entropy' or 'focal'
            class_weights: Optional weights for imbalanced classes
            device: 'cuda' or 'cpu'
            checkpoint_dir: Directory for model checkpoints
            log_dir: Directory for logs
            early_stopping_patience: Epochs to wait for improvement
            weight_decay: Optimizer weight decay
            scheduler_type: LR scheduler type
            scheduler_kwargs: Scheduler arguments
            checkpoint_monitor: Metric name to monitor for checkpointing
            save_best_only: Save only the best checkpoint
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = num_epochs
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir = Path(log_dir)
        
        # Create directories
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Optimizer
        if optimizer_type.lower() == "adam":
            self.optimizer = optim.Adam(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
            )
        elif optimizer_type.lower() == "sgd":
            self.optimizer = optim.SGD(
                model.parameters(),
                lr=learning_rate,
                momentum=0.9,
                weight_decay=weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")
        
        # Loss function
        self.loss_fn = get_loss_function(
            loss_type=loss_type,
            class_weights=class_weights,
            device=device,
        )
        
        # Callbacks
        self.early_stopping = EarlyStopping(patience=early_stopping_patience)
        self.checkpoint = ModelCheckpoint(
            checkpoint_dir=checkpoint_dir,
            monitor=checkpoint_monitor,
            save_best_only=save_best_only,
        )
        self.lr_scheduler = LRScheduler(scheduler_type=scheduler_type, **(scheduler_kwargs or {}))
        self.lr_scheduler.create_scheduler(self.optimizer)
        
        # Metrics
        num_classes = model.num_classes
        self.train_metrics = MetricsCalculator(num_classes)
        self.val_metrics = MetricsCalculator(num_classes)
        
        # Logging
        self.history = []
        self.log_file = self.log_dir / "training_history.csv"
        
        # Print info
        print(f"Device: {device}")
        print(f"Loss function: {loss_type}")
        print(f"Optimizer: {optimizer_type} (lr={learning_rate})")
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        self.train_metrics.reset()
        
        for batch_idx, (images, labels, _) in enumerate(self.train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.loss_fn(outputs, labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Update metrics
            self.train_metrics.update(outputs, labels, loss)
            
            # Progress
            if (batch_idx + 1) % max(1, len(self.train_loader) // 5) == 0:
                metrics = self.train_metrics.get_metrics()
                print(f"  Batch [{batch_idx+1}/{len(self.train_loader)}] Loss: {metrics['loss']:.4f}")
        
        return self.train_metrics.get_metrics()
    
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        self.val_metrics.reset()
        
        with torch.no_grad():
            for images, labels, _ in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.loss_fn(outputs, labels)
                
                self.val_metrics.update(outputs, labels, loss)
        
        return self.val_metrics.get_metrics()
    
    def train(self) -> Dict:
        """Main training loop."""
        print(f"\nStarting training for {self.num_epochs} epochs...")
        start_time = datetime.now()
        
        for epoch in range(1, self.num_epochs + 1):
            print(f"\nEpoch [{epoch}/{self.num_epochs}]")
            
            # Train
            train_metrics = self.train_epoch()
            print(f"  Train Loss: {train_metrics['loss']:.4f}, Accuracy: {train_metrics['accuracy']:.2f}%")
            
            # Validate
            val_metrics = self.validate_epoch()
            print(f"  Val Loss: {val_metrics['loss']:.4f}, Accuracy: {val_metrics['accuracy']:.2f}%")
            
            # Step scheduler
            if self.lr_scheduler.scheduler_type == "reduce_on_plateau":
                self.lr_scheduler.step(val_metrics.get("loss"))
            else:
                self.lr_scheduler.step()
            
            # Save latest checkpoint every epoch
            self.checkpoint.save_checkpoint(
                self.model,
                self.optimizer,
                epoch,
                val_metrics,
                filename=f"checkpoint_epoch_{epoch:03d}.pth",
                save_best_only=False,
            )

            # Save best model checkpoint
            if self.checkpoint.save_checkpoint(
                self.model,
                self.optimizer,
                epoch,
                val_metrics,
                filename="best_model.pth"
            ):
                print(f"  ✓ Best model saved")
            
            # Log history
            log_entry = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
            }
            self.history.append(log_entry)
            self._save_history()
            
            # Early stopping
            if self.early_stopping(val_metrics["accuracy"]):
                print(f"\nEarly stopping triggered at epoch {epoch}")
                break
        
        elapsed_time = datetime.now() - start_time
        print(f"\nTraining completed in {elapsed_time}")
        
        return {
            "history": self.history,
            "total_epochs": epoch,
            "training_time": str(elapsed_time),
        }
    
    def _save_history(self):
        """Save training history to CSV."""
        if not self.history:
            return
        
        with open(self.log_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.history[0].keys())
            writer.writeheader()
            writer.writerows(self.history)
