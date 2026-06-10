"""Loss functions for training."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    From "Focal Loss for Dense Object Detection"
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        """
        Args:
            alpha: Weighting factor in range (0,1) to balance classes
            gamma: Exponent for modulation term
            reduction: 'mean' or 'sum'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Logits of shape (N, C)
            targets: Labels of shape (N,)
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        p_t = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss
        
        if self.reduction == "mean":
            return focal_loss.mean()
        else:
            return focal_loss.sum()


def get_loss_function(
    loss_type: str = "cross_entropy",
    class_weights: Optional[torch.Tensor] = None,
    device: str = "cpu",
    **kwargs
) -> nn.Module:
    """
    Get loss function.
    
    Args:
        loss_type: Type of loss ('cross_entropy', 'focal')
        class_weights: Optional class weights for imbalanced data
        device: Device to place weights on
        **kwargs: Additional arguments for loss function
    
    Returns:
        Loss function module
    """
    if class_weights is not None:
        class_weights = class_weights.to(device)
    
    if loss_type == "cross_entropy":
        return nn.CrossEntropyLoss(weight=class_weights, reduction="mean")
    elif loss_type == "focal":
        return FocalLoss(
            alpha=kwargs.get("alpha", 0.25),
            gamma=kwargs.get("gamma", 2.0),
            reduction="mean"
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
