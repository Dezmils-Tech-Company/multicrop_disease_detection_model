#!/usr/bin/env python
"""Quick training test on small dataset subset."""
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data import create_dataloaders, CropDiseaseDataset
from models import DiseaseDetectionModel
from training import Trainer


def quick_train_test():
    """Run a quick training test."""
    print("\n" + "="*70)
    print("QUICK TRAINING TEST (1 epoch on small batch)")
    print("="*70)
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")
    
    # Create dataloaders with small batch for testing
    print("\nCreating dataloaders...")
    train_loader, val_loader, test_loader, class_to_idx = create_dataloaders(
        train_dir="dataset/train",
        val_dir="dataset/val",
        batch_size=16,  # Small batch for quick test
        num_workers=0,
        image_size=224,
    )
    
    # Get class weights
    train_dataset = CropDiseaseDataset(
        "dataset/train",
        class_to_idx=class_to_idx,
    )
    class_weights = train_dataset.get_class_weights()
    
    # Create model
    print("\nCreating model...")
    model = DiseaseDetectionModel(
        num_classes=len(class_to_idx),
        model_type="resnet50",
        pretrained=True,
        freeze_backbone=0,
        head_hidden_dims=[512],
        dropout_rate=0.3,
    )
    
    # Create trainer
    print("\nCreating trainer...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=1,  # Just 1 epoch for testing
        learning_rate=1e-4,
        optimizer_type="adam",
        loss_type="cross_entropy",
        class_weights=class_weights,
        device=device,
        checkpoint_dir="checkpoints",
        log_dir="logs",
        early_stopping_patience=3,
    )
    
    # Train for 1 epoch
    print("\nTraining for 1 epoch...")
    results = trainer.train()
    
    print("\n" + "="*70)
    print("TRAINING TEST RESULTS")
    print("="*70)
    print(f"Completed epochs: {results['total_epochs']}")
    print(f"Training time: {results['training_time']}")
    
    if results['history']:
        latest = results['history'][-1]
        print(f"\nFinal metrics:")
        print(f"  Train Loss: {latest['train_loss']:.4f}")
        print(f"  Train Accuracy: {latest['train_accuracy']:.2f}%")
        print(f"  Val Loss: {latest['val_loss']:.4f}")
        print(f"  Val Accuracy: {latest['val_accuracy']:.2f}%")
    
    print("\n[OK] Training test completed successfully!")
    return True


if __name__ == "__main__":
    try:
        success = quick_train_test()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
