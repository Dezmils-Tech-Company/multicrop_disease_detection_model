"""DataLoader utilities for efficient batch loading."""
from typing import Dict, Tuple, Optional
from torch.utils.data import DataLoader
from .dataset import CropDiseaseDataset
from .transforms import get_train_transforms, get_val_transforms


def create_dataloaders(
    train_dir: str,
    val_dir: str,
    test_dir: Optional[str] = None,
    batch_size: int = 64,
    num_workers: int = 4,
    image_size: int = 224,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader], Dict[str, int]]:
    """
    Create train, validation, and optionally test dataloaders.
    
    Args:
        train_dir: Path to training data directory
        val_dir: Path to validation data directory
        test_dir: Path to test data directory (optional)
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes
        image_size: Image size for transforms
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader, class_to_idx)
    """
    # Build class mapping from training data
    train_dataset = CropDiseaseDataset(
        train_dir,
        transform=get_train_transforms(image_size),
    )
    class_to_idx = train_dataset.class_to_idx
    num_classes = len(class_to_idx)
    
    # Create datasets with shared class mapping
    val_dataset = CropDiseaseDataset(
        val_dir,
        transform=get_val_transforms(image_size),
        class_to_idx=class_to_idx,
    )
    
    test_dataset = None
    if test_dir:
        test_dataset = CropDiseaseDataset(
            test_dir,
            transform=get_val_transforms(image_size),
            class_to_idx=class_to_idx,
        )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    test_loader = None
    if test_dataset:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
    
    print(f"Created dataloaders:")
    print(f"  Train: {len(train_dataset)} images")
    print(f"  Val: {len(val_dataset)} images")
    if test_loader:
        print(f"  Test: {len(test_dataset)} images")
    print(f"  Classes: {num_classes}")
    print(f"  Batch size: {batch_size}")
    
    return train_loader, val_loader, test_loader, class_to_idx


def get_class_weights_from_loader(train_loader: DataLoader):
    """Extract class weights from training loader for imbalanced classes."""
    return train_loader.dataset.get_class_weights()
