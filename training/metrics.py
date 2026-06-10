"""Metrics calculation and evaluation."""
import torch
import torch.nn.functional as F
from typing import Dict, Tuple
import numpy as np


class MetricsCalculator:
    """Calculate training and validation metrics."""
    
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.reset()
    
    def reset(self):
        """Reset metrics."""
        self.total = 0
        self.correct = 0
        self.top5_correct = 0
        self.per_class_correct = np.zeros(self.num_classes)
        self.per_class_total = np.zeros(self.num_classes)
        self.loss_sum = 0.0
        self.loss_count = 0
    
    def update(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        loss: torch.Tensor = None,
    ):
        """
        Update metrics with batch results.
        
        Args:
            outputs: Model outputs (logits) of shape (N, C)
            targets: Ground truth labels of shape (N,)
            loss: Batch loss value
        """
        batch_size = targets.size(0)
        
        # Top-1 accuracy
        _, predictions = outputs.max(1)
        correct = predictions.eq(targets).sum().item()
        self.total += batch_size
        self.correct += correct
        
        # Top-5 accuracy
        if self.num_classes >= 5:
            _, top5_pred = outputs.topk(5, 1, largest=True, sorted=True)
            correct_top5 = top5_pred.eq(targets.view(-1, 1).expand_as(top5_pred)).any(1).sum().item()
            self.top5_correct += correct_top5
        
        # Per-class metrics
        for i in range(self.num_classes):
            mask = targets == i
            if mask.sum() > 0:
                class_correct = predictions[mask].eq(targets[mask]).sum().item()
                self.per_class_correct[i] += class_correct
                self.per_class_total[i] += mask.sum().item()
        
        # Loss
        if loss is not None:
            self.loss_sum += loss.item() * batch_size
            self.loss_count += batch_size
    
    def get_metrics(self) -> Dict[str, float]:
        """Get current metrics."""
        metrics = {
            "loss": self.loss_sum / self.loss_count if self.loss_count > 0 else 0.0,
            "accuracy": 100.0 * self.correct / self.total if self.total > 0 else 0.0,
        }
        
        if self.num_classes >= 5 and self.total > 0:
            metrics["top5_accuracy"] = 100.0 * self.top5_correct / self.total
        
        # Per-class accuracy
        per_class_acc = []
        for i in range(self.num_classes):
            if self.per_class_total[i] > 0:
                acc = 100.0 * self.per_class_correct[i] / self.per_class_total[i]
                per_class_acc.append(acc)
        
        if per_class_acc:
            metrics["macro_accuracy"] = np.mean(per_class_acc)
        
        return metrics
    
    def get_per_class_metrics(self) -> Dict[int, Dict[str, float]]:
        """Get per-class accuracy."""
        per_class = {}
        for i in range(self.num_classes):
            if self.per_class_total[i] > 0:
                acc = 100.0 * self.per_class_correct[i] / self.per_class_total[i]
                per_class[i] = {"accuracy": acc, "count": int(self.per_class_total[i])}
        return per_class


def compute_confusion_matrix(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
) -> np.ndarray:
    """
    Compute confusion matrix.
    
    Args:
        outputs: Model predictions (logits)
        targets: Ground truth labels
        num_classes: Number of classes
    
    Returns:
        Confusion matrix of shape (num_classes, num_classes)
    """
    _, predictions = outputs.max(1)
    predictions = predictions.cpu().numpy()
    targets = targets.cpu().numpy()
    
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for pred, target in zip(predictions, targets):
        cm[target, pred] += 1
    
    return cm
