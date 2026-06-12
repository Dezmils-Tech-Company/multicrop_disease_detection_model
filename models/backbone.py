"""Pretrained backbone models."""
import torch
import torch.nn as nn
from torchvision import models
from typing import Optional, Tuple


def load_backbone(
    model_type: str = "resnet50",
    pretrained: bool = True,
    freeze_layers: int = 0,
) -> Tuple[nn.Module, int]:
    """
    Load pretrained backbone model.
    
    Args:
        model_type: Type of backbone ('resnet50', 'resnet152', 'efficientnet_b4', etc.)
        pretrained: Whether to load ImageNet pretrained weights
        freeze_layers: Number of initial residual blocks to freeze
    
    Returns:
        Tuple of (model, output_features) where output_features is the
        number of features from the backbone before final classification layer
    """
    if model_type == "resnet50":
        model = models.resnet50(pretrained=pretrained)
        output_features = 2048
    elif model_type == "resnet152":
        model = models.resnet152(pretrained=pretrained)
        output_features = 2048
    elif model_type == "efficientnet_b4":
        model = models.efficientnet_b4(pretrained=pretrained)
        output_features = 1792
    elif model_type == "densenet121":
        model = models.densenet121(pretrained=pretrained)
        output_features = 1024
    elif model_type == "vit_b_16":
        model = models.vit_b_16(pretrained=pretrained)
        output_features = 768
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Remove classification layer
    if hasattr(model, 'fc'):
        # ResNet models
        model.fc = nn.Identity()
    elif hasattr(model, 'classifier'):
        # DenseNet, EfficientNet models
        model.classifier = nn.Identity()
    elif hasattr(model, 'heads'):
        # Vision Transformer
        model.heads = nn.Identity()
    
    # Freeze initial layers
    if freeze_layers > 0:
        freeze_backbone_layers(model, model_type, freeze_layers)
    
    return model, output_features


def freeze_backbone_layers(model: nn.Module, model_type: str, num_blocks: int):
    """Freeze initial layers of backbone for fine-tuning."""
    params_to_freeze = []
    
    if "resnet" in model_type:
        if num_blocks >= 1:
            for param in model.conv1.parameters():
                param.requires_grad = False
            for param in model.bn1.parameters():
                param.requires_grad = False
        if num_blocks >= 2:
            for param in model.layer1.parameters():
                param.requires_grad = False
        if num_blocks >= 3:
            for param in model.layer2.parameters():
                param.requires_grad = False
        if num_blocks >= 4:
            for param in model.layer3.parameters():
                param.requires_grad = False
    
    elif "densenet" in model_type:
        for i in range(num_blocks):
            if hasattr(model, f'denseblock{i+1}'):
                for param in getattr(model, f'denseblock{i+1}').parameters():
                    param.requires_grad = False
    
    elif "efficientnet" in model_type:
        layers = list(model.features.children())
        for i in range(min(num_blocks, len(layers))):
            for param in layers[i].parameters():
                param.requires_grad = False
    
    print(f"Froze {num_blocks} initial blocks in {model_type}")


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Count trainable and total parameters.
    
    Returns:
        Tuple of (trainable_params, total_params)
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total
