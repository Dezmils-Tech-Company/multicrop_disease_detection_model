"""Main training loop."""
import os
import time
import signal
import threading
import traceback
import random
import json
import csv
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Tuple, Optional

from torch.cuda import amp

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
        num_epochs: int = 25,
        learning_rate: float = 1e-3,
        optimizer_type: str = "adam",
        loss_type: str = "cross_entropy",
        class_weights: Optional[torch.Tensor] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        checkpoint_dir: str = "checkpoints",
        log_dir: str = "logs",
        early_stopping_patience: int = 7,
        weight_decay: float = 0.0,
        scheduler_type: str = "cosine",
        scheduler_kwargs: Optional[dict] = None,
        checkpoint_monitor: str = "val_accuracy",
        save_best_only: bool = True,
        resume_from: Optional[str] = None,
        use_amp: bool = False,
        autosave_seconds: int = 300,
        max_autosaves: int = 5,
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
        self.resume_from = resume_from
        self.use_amp = use_amp
        self.autosave_seconds = max(5, int(autosave_seconds))
        self.max_autosaves = max(1, int(max_autosaves))
        self._autosave_thread = None
        self._autosave_stop = threading.Event()
        
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
        # AMP
        self.scaler = amp.GradScaler() if self.use_amp else None

        # Performance tweaks
        try:
            if torch.backends.cudnn.is_available():
                torch.backends.cudnn.benchmark = True
        except Exception:
            pass

        # Register signal handlers to save on interruption
        try:
            signal.signal(signal.SIGINT, self._handle_signal)
            signal.signal(signal.SIGTERM, self._handle_signal)
        except Exception:
            # Signal handling may not be available on some platforms
            pass

        # If resume requested, attempt to load
        self.start_epoch = 1
        if self.resume_from:
            try:
                loaded_epoch = self._load_full_checkpoint(self.resume_from)
                if isinstance(loaded_epoch, int):
                    self.start_epoch = loaded_epoch + 1
                    print(f"Resuming from checkpoint at epoch {loaded_epoch} -> starting at {self.start_epoch}")
            except Exception as e:
                print(f"Failed to resume from {self.resume_from}: {e}")
        
        # Print info
        print(f"Device: {device}")
        print(f"Loss function: {loss_type}")
        print(f"Optimizer: {optimizer_type} (lr={learning_rate})")

    # ---- Checkpointing & resume helpers ----
    def _gather_rng_states(self) -> Dict:
        return {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }

    def _save_full_checkpoint(self, epoch: int, metrics: Dict, filename: Optional[str] = None) -> str:
        """Save an extended checkpoint including optimizer, scheduler, scaler, RNG and history."""
        if filename is None:
            filename = f"full_checkpoint_epoch_{epoch:03d}.pth"

        path = self.checkpoint_dir / filename
        data = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "history": self.history,
            "rng_states": self._gather_rng_states(),
            "early_stopping": {
                "best_value": self.early_stopping.best_value,
                "counter": self.early_stopping.counter,
                "should_stop": self.early_stopping.should_stop,
            },
        }
        # scheduler state
        try:
            if hasattr(self.lr_scheduler, "scheduler") and self.lr_scheduler.scheduler is not None:
                data["scheduler_state_dict"] = self.lr_scheduler.scheduler.state_dict()
        except Exception:
            pass

        # scaler for AMP
        if self.scaler is not None:
            try:
                data["scaler_state_dict"] = self.scaler.state_dict()
            except Exception:
                pass

        torch.save(data, path)
        # Manage autosave files if applicable
        if filename.startswith("autosave_"):
            self._prune_autosaves()

        return str(path)

    def _load_full_checkpoint(self, checkpoint_path: str) -> Optional[int]:
        """Load extended checkpoint and restore state. Returns epoch loaded."""
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(checkpoint_path)

        checkpoint = torch.load(path, map_location=self.device)
        # load model
        try:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        except Exception:
            # try strict=False for shape mismatches
            self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)

        # optimizer
        try:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        except Exception:
            pass

        # scheduler
        try:
            if "scheduler_state_dict" in checkpoint and hasattr(self.lr_scheduler, "scheduler"):
                self.lr_scheduler.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        except Exception:
            pass

        # scaler
        try:
            if "scaler_state_dict" in checkpoint and self.scaler is not None:
                self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        except Exception:
            pass

        # history
        try:
            if "history" in checkpoint:
                self.history = checkpoint["history"]
        except Exception:
            pass

        # rng states
        try:
            rng = checkpoint.get("rng_states")
            if rng:
                random.setstate(rng.get("python"))
                np.random.set_state(rng.get("numpy"))
                torch.set_rng_state(rng.get("torch"))
                if torch.cuda.is_available() and rng.get("cuda") is not None:
                    torch.cuda.set_rng_state_all(rng.get("cuda"))
        except Exception:
            pass

        # early stopping
        try:
            es = checkpoint.get("early_stopping")
            if es:
                self.early_stopping.best_value = es.get("best_value")
                self.early_stopping.counter = es.get("counter", 0)
                self.early_stopping.should_stop = es.get("should_stop", False)
        except Exception:
            pass

        return checkpoint.get("epoch")

    def _prune_autosaves(self):
        files = sorted(self.checkpoint_dir.glob("autosave_*.pth"), key=os.path.getmtime)
        while len(files) > self.max_autosaves:
            try:
                files[0].unlink()
            except Exception:
                pass
            files = files[1:]

    def _autosave_worker(self):
        while not self._autosave_stop.wait(self.autosave_seconds):
            try:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                fname = f"autosave_{ts}.pth"
                # save latest epoch if available else 0
                current_epoch = getattr(self, "_current_epoch", 0)
                metrics = self.val_metrics.get_metrics() if hasattr(self, "val_metrics") else {}
                self._save_full_checkpoint(current_epoch, metrics, filename=fname)
                print(f"Autosaved checkpoint: {fname}")
            except Exception:
                traceback.print_exc()

    def _start_autosave_thread(self):
        if self._autosave_thread is None or not self._autosave_thread.is_alive():
            self._autosave_stop.clear()
            self._autosave_thread = threading.Thread(target=self._autosave_worker, daemon=True)
            self._autosave_thread.start()

    def _stop_autosave_thread(self):
        if self._autosave_thread and self._autosave_thread.is_alive():
            self._autosave_stop.set()
            self._autosave_thread.join(timeout=2.0)

    def _handle_signal(self, signum, frame):
        print(f"Received signal {signum}, saving checkpoint and exiting gracefully...")
        try:
            epoch = getattr(self, "_current_epoch", 0)
            metrics = self.val_metrics.get_metrics() if hasattr(self, "val_metrics") else {}
            self._save_full_checkpoint(epoch, metrics, filename=f"interrupted_epoch_{epoch:03d}.pth")
        except Exception:
            traceback.print_exc()
        finally:
            os._exit(0)
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        self.train_metrics.reset()
        for batch_idx, (images, labels, _) in enumerate(self.train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Forward + backward with optional AMP
            if self.use_amp and self.scaler is not None:
                with amp.autocast():
                    outputs = self.model(images)
                    loss = self.loss_fn(outputs, labels)

                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                # Unscale before clipping
                try:
                    self.scaler.unscale_(self.optimizer)
                except Exception:
                    pass
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.loss_fn(outputs, labels)
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
        print(f"\nStarting training for {self.num_epochs} epochs... (start_epoch={self.start_epoch})")
        start_time = datetime.now()
        # Start autosave thread
        self._start_autosave_thread()

        try:
            epoch_range = range(self.start_epoch, self.num_epochs + 1)
            for epoch in epoch_range:
                self._current_epoch = epoch
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
            
                # Save latest checkpoint every epoch (lightweight)
                self.checkpoint.save_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch,
                    val_metrics,
                    filename=f"checkpoint_epoch_{epoch:03d}.pth",
                    save_best_only=False,
                )

                # Save best model checkpoint (lightweight)
                if self.checkpoint.save_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch,
                    val_metrics,
                    filename="best_model.pth"
                ):
                    print(f"  ✓ Best model saved")

                # Save extended full checkpoint (robust resume)
                try:
                    self._save_full_checkpoint(epoch, val_metrics)
                except Exception:
                    traceback.print_exc()
            
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
        except Exception:
            # On unexpected exception, save a checkpoint and re-raise
            try:
                epoch = getattr(self, "_current_epoch", 0)
                metrics = locals().get("val_metrics", {}) or {}
                self._save_full_checkpoint(epoch, metrics, filename=f"crash_epoch_{epoch:03d}.pth")
                print("Saved crash checkpoint.")
            except Exception:
                traceback.print_exc()
            raise
        finally:
            # Stop autosave thread and ensure final checkpoint
            try:
                self._stop_autosave_thread()
                final_epoch = getattr(self, "_current_epoch", self.num_epochs)
                final_metrics = getattr(self, "val_metrics", {}).get_metrics() if hasattr(self, "val_metrics") else {}
                self._save_full_checkpoint(final_epoch, final_metrics, filename=f"final_epoch_{final_epoch:03d}.pth")
            except Exception:
                traceback.print_exc()
        elapsed_time = datetime.now() - start_time
        print(f"\nTraining completed in {elapsed_time}")

        return {
            "history": self.history,
            "total_epochs": getattr(self, "_current_epoch", self.num_epochs),
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