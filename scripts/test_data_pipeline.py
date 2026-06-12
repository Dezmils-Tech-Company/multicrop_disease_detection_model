#!/usr/bin/env python
"""Test data pipeline for loading and preprocessing."""
import torch
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data import create_dataloaders, CropDiseaseDataset
from data.transforms import get_train_transforms, get_val_transforms


def test_dataset_loading():
    """Test loading a single dataset."""
    print("\n" + "="*70)
    print("TEST 1: Dataset Loading")
    print("="*70)
    
    # Load training dataset
    dataset = CropDiseaseDataset(
        "dataset/train",
        transform=get_train_transforms(),
    )
    
    print(f"\nDataset Info:")
    print(f"  Total images: {len(dataset)}")
    print(f"  Number of classes: {len(dataset.classes)}")
    print(f"  First 5 classes: {dataset.classes[:5]}")
    
    # Test loading a sample
    image, label, class_name = dataset[0]
    print(f"\nSample loaded:")
    print(f"  Image shape: {image.shape}")
    print(f"  Image dtype: {image.dtype}")
    print(f"  Label: {label}")
    print(f"  Class name: {class_name}")
    
    # Get class distribution
    class_dist = dataset.get_class_distribution()
    print(f"\nClass distribution (first 5):")
    for i, (cls, count) in enumerate(list(class_dist.items())[:5]):
        print(f"  {cls}: {count}")
    
    # Get class weights
    weights = dataset.get_class_weights()
    print(f"\nClass weights:")
    print(f"  Shape: {weights.shape}")
    print(f"  Min: {weights.min():.4f}, Max: {weights.max():.4f}")
    print(f"  Mean: {weights.mean():.4f}")
    
    return True


def test_dataloaders():
    """Test creating and iterating dataloaders."""
    print("\n" + "="*70)
    print("TEST 2: DataLoader Creation and Iteration")
    print("="*70)
    
    # Create dataloaders
    train_loader, val_loader, test_loader, class_to_idx = create_dataloaders(
        train_dir="dataset/train",
        val_dir="dataset/val",
        test_dir="dataset/test",
        batch_size=32,
        num_workers=0,  # Set to 0 for testing
        image_size=224,
    )
    
    # Test train loader
    print("\nIterating through training batches...")
    for batch_idx, (images, labels, class_names) in enumerate(train_loader):
        print(f"\nBatch {batch_idx}:")
        print(f"  Images shape: {images.shape}")
        print(f"  Labels shape: {labels.shape}")
        print(f"  Labels dtype: {labels.dtype}")
        print(f"  Labels range: [{labels.min()}, {labels.max()}]")
        print(f"  Batch classes: {set(class_names)}")
        print(f"  Image stats - min: {images.min():.3f}, max: {images.max():.3f}, mean: {images.mean():.3f}")
        
        if batch_idx >= 1:  # Just test 2 batches
            break
    
    print(f"\n✓ DataLoaders created and tested successfully")
    return True


def test_transforms():
    """Test data transforms."""
    print("\n" + "="*70)
    print("TEST 3: Data Transforms")
    print("="*70)
    
    from PIL import Image
    import numpy as np
    
    # Load a real image
    dataset = CropDiseaseDataset("dataset/train", transform=None)
    img_path, _ = dataset.samples[0]
    
    print(f"\nOriginal image: {img_path}")
    
    # Load with PIL
    img_pil = Image.open(img_path)
    print(f"  Size: {img_pil.size}")
    print(f"  Mode: {img_pil.mode}")
    
    # Apply train transforms
    train_transform = get_train_transforms(224)
    img_tensor_train = train_transform(img_pil)
    print(f"\nTrain transform output:")
    print(f"  Shape: {img_tensor_train.shape}")
    print(f"  Dtype: {img_tensor_train.dtype}")
    print(f"  Min: {img_tensor_train.min():.3f}, Max: {img_tensor_train.max():.3f}")
    
    # Apply val transforms
    val_transform = get_val_transforms(224)
    img_tensor_val = val_transform(img_pil)
    print(f"\nVal transform output:")
    print(f"  Shape: {img_tensor_val.shape}")
    print(f"  Dtype: {img_tensor_val.dtype}")
    print(f"  Min: {img_tensor_val.min():.3f}, Max: {img_tensor_val.max():.3f}")
    
    print(f"\n✓ Transforms working correctly")
    return True


def main():
    """Run all tests."""
    print("\n" + "#"*70)
    print("# DATA PIPELINE TESTING")
    print("#"*70)
    
    try:
        test_dataset_loading()
        test_transforms()
        test_dataloaders()
        
        print("\n" + "="*70)
        print("ALL TESTS PASSED ✓")
        print("="*70)
        print("\nData pipeline is ready for training!")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
