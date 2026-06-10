"""Training callbacks for checkpointing and scheduling."""
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional


class EarlyStopping:
    """Stop training if validation metric doesn't improve."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        """
        Args:
            patience: Number of epochs with no improvement to wait
            min_delta: Minimum change in monitored metric to qualify as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_value = None
        self.should_stop = False
    
    def __call__(self, current_value: float) -> bool:
        """
        Check if training should stop.
        
        Returns:
            True if training should stop
        """
        if self.best_value is None:
            self.best_value = current_value
        elif current_value > self.best_value + self.min_delta:
            self.best_value = current_value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        
        return self.should_stop


class ModelCheckpoint:
    """Save model checkpoint when metric improves."""
    
    def __init__(self, checkpoint_dir: str, monitor: str = "val_accuracy", save_best_only: bool = True):
        """
        Args:
            checkpoint_dir: Directory to save checkpoints
            monitor: Metric to monitor
            save_best_only: Only save if metric improves
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.best_value = None
    
    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: dict,
        filename: str = "checkpoint.pth"
    ) -> bool:
        """
        Save checkpoint.
        
        Returns:
            True if checkpoint was saved
        """
        current_value = metrics.get(self.monitor)
        
        should_save = False
        if not self.save_best_only:
            should_save = True
        elif self.best_value is None or current_value > self.best_value:
            should_save = True
            self.best_value = current_value
        
        if should_save:
            checkpoint_path = self.checkpoint_dir / filename
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "metrics": metrics,
            }, checkpoint_path)
            return True
        
        return False
    
    def load_checkpoint(self, model: nn.Module, optimizer: torch.optim.Optimizer, checkpoint_path: str):
        """Load checkpoint."""
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch = checkpoint["epoch"]
        return epoch


class LRScheduler:
    """Learning rate scheduler."""
    
    def __init__(self, scheduler_type: str = "cosine", **kwargs):
        """
        Args:
            scheduler_type: 'cosine', 'step', or 'exponential'
            **kwargs: Arguments for scheduler
        """
        self.scheduler_type = scheduler_type
        self.kwargs = kwargs
        self.scheduler = None
    
    def create_scheduler(self, optimizer: torch.optim.Optimizer):
        """Create scheduler for optimizer."""
        if self.scheduler_type == "cosine":
            from torch.optim.lr_scheduler import CosineAnnealingLR
            self.scheduler = CosineAnnealingLR(
                optimizer,
                T_max=self.kwargs.get("T_max", 100),
                eta_min=self.kwargs.get("eta_min", 0)
            )
        elif self.scheduler_type == "step":
            from torch.optim.lr_scheduler import StepLR
            self.scheduler = StepLR(
                optimizer,
                step_size=self.kwargs.get("step_size", 10),
                gamma=self.kwargs.get("gamma", 0.1)
            )
        elif self.scheduler_type == "exponential":
            from torch.optim.lr_scheduler import ExponentialLR
            self.scheduler = ExponentialLR(
                optimizer,
                gamma=self.kwargs.get("gamma", 0.9)
            )
        else:
            raise ValueError(f"Unknown scheduler type: {self.scheduler_type}")
    
    def step(self):
        """Step scheduler."""
        if self.scheduler:
            self.scheduler.step()
