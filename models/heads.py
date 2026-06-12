"""Classification head architectures."""
import torch
import torch.nn as nn
from typing import Optional


class ClassificationHead(nn.Module):
    """
    Classification head for disease detection.
    
    Maps backbone features to class predictions with optional dropout
    and layer normalization for better training stability.
    """
    
    def __init__(
        self,
        input_features: int,
        num_classes: int,
        hidden_dims: Optional[list] = None,
        dropout_rate: float = 0.3,
    ):
        """
        Args:
            input_features: Number of input features from backbone
            num_classes: Number of output classes
            hidden_dims: Optional list of hidden layer dimensions
            dropout_rate: Dropout probability
        """
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [512]
        
        layers = []
        in_dim = input_features
        
        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            in_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(in_dim, num_classes))
        
        self.head = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, input_features)
        
        Returns:
            Logits of shape (batch_size, num_classes)
        """
        return self.head(x)
