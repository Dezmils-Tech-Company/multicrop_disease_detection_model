"""Main disease detection model combining backbone and heads."""
import torch
import torch.nn as nn
from typing import Optional, Dict
from .backbone import load_backbone, count_parameters
from .heads import ClassificationHead


class DiseaseDetectionModel(nn.Module):
    """
    End-to-end disease detection model.
    
    Combines pretrained backbone with custom classification head
    for crop disease detection.
    """
    
    def __init__(
        self,
        num_classes: int,
        model_type: str = "resnet50",
        pretrained: bool = True,
        freeze_backbone: int = 0,
        head_hidden_dims: Optional[list] = None,
        dropout_rate: float = 0.3,
    ):
        """
        Args:
            num_classes: Number of disease classes to predict
            model_type: Backbone architecture type
            pretrained: Use ImageNet pretrained weights
            freeze_backbone: Number of blocks to freeze
            head_hidden_dims: Hidden dimensions for classification head
            dropout_rate: Dropout rate in head
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.model_type = model_type
        
        # Load backbone
        self.backbone, backbone_features = load_backbone(
            model_type=model_type,
            pretrained=pretrained,
            freeze_layers=freeze_backbone,
        )
        
        # Add classification head
        self.head = ClassificationHead(
            input_features=backbone_features,
            num_classes=num_classes,
            hidden_dims=head_hidden_dims,
            dropout_rate=dropout_rate,
        )
        
        # Store config
        self.config = {
            "num_classes": num_classes,
            "model_type": model_type,
            "pretrained": pretrained,
            "freeze_backbone": freeze_backbone,
            "head_hidden_dims": head_hidden_dims or [512],
            "dropout_rate": dropout_rate,
        }
        
        # Print model info
        self._print_model_info()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through backbone and head.
        
        Args:
            x: Input tensor of shape (batch_size, 3, H, W)
        
        Returns:
            Logits of shape (batch_size, num_classes)
        """
        # Backbone forward pass
        features = self.backbone(x)
        
        # Global average pooling (for models like ResNet)
        if features.dim() == 4:
            features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
            features = features.flatten(1)
        
        # Classification head
        logits = self.head(features)
        return logits
    
    def _print_model_info(self):
        """Print model architecture information."""
        trainable, total = count_parameters(self)
        print(f"\nModel: {self.model_type}")
        print(f"Total parameters: {total:,}")
        print(f"Trainable parameters: {trainable:,}")
        print(f"Output classes: {self.num_classes}")
    
    def freeze_backbone(self):
        """Freeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("Backbone frozen")
    
    def unfreeze_backbone(self):
        """Unfreeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("Backbone unfrozen")
    
    def get_config(self) -> Dict:
        """Get model configuration."""
        return self.config.copy()
