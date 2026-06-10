#!/usr/bin/env python
"""Test model architecture and forward pass."""
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models import DiseaseDetectionModel


def test_model_creation():
    """Test model initialization."""
    print("\n" + "="*70)
    print("TEST 1: Model Creation")
    print("="*70)
    
    # Create model
    model = DiseaseDetectionModel(
        num_classes=53,
        model_type="resnet50",
        pretrained=True,
        freeze_backbone=0,
        head_hidden_dims=[512],
        dropout_rate=0.3,
    )
    
    print("\n✓ Model created successfully")
    return model


def test_forward_pass(model):
    """Test forward pass."""
    print("\n" + "="*70)
    print("TEST 2: Forward Pass")
    print("="*70)
    
    # Create dummy input
    batch_size = 8
    dummy_input = torch.randn(batch_size, 3, 224, 224)
    print(f"\nInput shape: {dummy_input.shape}")
    
    # Forward pass
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
    print(f"Output dtype: {output.dtype}")
    
    # Check softmax probabilities
    probs = torch.nn.functional.softmax(output, dim=1)
    print(f"\nProbabilities:")
    print(f"  Min: {probs.min():.4f}, Max: {probs.max():.4f}")
    print(f"  Sum per sample: {probs.sum(dim=1).mean():.4f} (should be 1.0)")
    
    # Check predictions
    predictions = output.argmax(dim=1)
    print(f"\nTop predictions: {predictions[:4].tolist()}")
    print(f"Prediction range: [{predictions.min()}, {predictions.max()}]")
    
    print("\n✓ Forward pass successful")
    return output


def test_loss_computation(model):
    """Test loss computation."""
    print("\n" + "="*70)
    print("TEST 3: Loss Computation")
    print("="*70)
    
    from training import get_loss_function
    from data import CropDiseaseDataset
    
    # Create loss functions
    print("\nTesting different loss functions...")
    
    # Cross-entropy loss
    ce_loss = get_loss_function("cross_entropy", device="cpu")
    print(f"  ✓ Cross-Entropy Loss created")
    
    # Focal loss
    focal_loss = get_loss_function("focal", device="cpu")
    print(f"  ✓ Focal Loss created")
    
    # Test with dummy batch
    dummy_output = torch.randn(16, 53)
    dummy_targets = torch.randint(0, 53, (16,))
    
    ce_value = ce_loss(dummy_output, dummy_targets)
    print(f"\nCross-Entropy loss value: {ce_value:.4f}")
    
    focal_value = focal_loss(dummy_output, dummy_targets)
    print(f"Focal loss value: {focal_value:.4f}")
    
    # Test with class weights
    dataset = CropDiseaseDataset("dataset/train")
    class_weights = dataset.get_class_weights()
    weighted_loss = get_loss_function("cross_entropy", class_weights=class_weights, device="cpu")
    weighted_value = weighted_loss(dummy_output, dummy_targets)
    print(f"Weighted Cross-Entropy loss: {weighted_value:.4f}")
    
    print("\n✓ Loss computation successful")


def test_model_freeze():
    """Test freezing backbone layers."""
    print("\n" + "="*70)
    print("TEST 4: Model Freezing")
    print("="*70)
    
    # Create model with frozen layers
    model = DiseaseDetectionModel(
        num_classes=53,
        model_type="resnet50",
        pretrained=True,
        freeze_backbone=2,  # Freeze 2 blocks
    )
    
    # Count trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")
    print(f"Frozen: {frozen_params:,} ({100*frozen_params/total_params:.1f}%)")
    
    # Test unfreezing
    model.unfreeze_backbone()
    trainable_params_unfrozen = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nAfter unfreezing:")
    print(f"  Trainable: {trainable_params_unfrozen:,} ({100*trainable_params_unfrozen/total_params:.1f}%)")
    
    print("\n✓ Model freezing successful")


def main():
    """Run all tests."""
    print("\n" + "#"*70)
    print("# MODEL ARCHITECTURE TESTING")
    print("#"*70)
    
    try:
        model = test_model_creation()
        test_forward_pass(model)
        test_loss_computation(model)
        test_model_freeze()
        
        print("\n" + "="*70)
        print("ALL TESTS PASSED ✓")
        print("="*70)
        print("\nModel architecture is ready for training!")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
