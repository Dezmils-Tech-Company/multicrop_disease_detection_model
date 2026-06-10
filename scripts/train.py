#!/usr/bin/env python
"""Main training script to train the crop disease detection model."""
import argparse
import json
import torch
import yaml
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data import create_dataloaders, CropDiseaseDataset
from models import DiseaseDetectionModel
from training import Trainer


def load_config(config_path: str) -> dict:
    """Load training configuration from YAML."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def main(args):
    """Main training pipeline."""
    # Load config
    config = load_config(args.config)
    print(f"Loaded config from {args.config}")
    print(json.dumps(config, indent=2))
    
    # Set device
    device = "cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    print(f"\nUsing device: {device}")
    
    # Create dataloaders
    print("\n" + "="*50)
    print("Creating data loaders...")
    print("="*50)
    
    train_loader, val_loader, test_loader, class_to_idx = create_dataloaders(
        train_dir=config["data"]["train_path"],
        val_dir=config["data"]["val_path"],
        test_dir=config["data"].get("test_path"),
        batch_size=config["training"]["batch_size"],
        num_workers=config["data"]["num_workers"],
        image_size=config["model"]["input_size"],
        image_extensions=tuple(config["data"].get("image_extensions", [".jpg", ".jpeg", ".png"])),
    )
    
    # Get class weights for imbalanced data
    train_dataset = CropDiseaseDataset(
        config["data"]["train_path"],
        class_to_idx=class_to_idx,
    )
    class_weights = train_dataset.get_class_weights()
    print(f"Class weights computed (min={class_weights.min():.3f}, max={class_weights.max():.3f})")
    
    # Create model
    print("\n" + "="*50)
    print("Creating model...")
    print("="*50)
    
    model = DiseaseDetectionModel(
        num_classes=len(class_to_idx),
        model_type=config["model"]["type"],
        pretrained=config["model"]["pretrained"],
        freeze_backbone=config["model"]["freeze_backbone_layers"],
        head_hidden_dims=[512],
        dropout_rate=0.3,
    )
    
    # Create trainer
    print("\n" + "="*50)
    print("Setting up trainer...")
    print("="*50)
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config["training"]["num_epochs"],
        learning_rate=config["training"]["learning_rate"],
        optimizer_type=config["training"]["optimizer"],
        loss_type=config["loss"]["type"],
        class_weights=class_weights,
        device=device,
        checkpoint_dir=config["callbacks"]["checkpoint_dir"],
        log_dir=config["callbacks"]["log_dir"],
        early_stopping_patience=config["callbacks"]["early_stopping_patience"],
        weight_decay=config["training"].get("weight_decay", 0.0),
        scheduler_type=config.get("scheduler", {}).get("type", "cosine"),
        scheduler_kwargs={
            "T_max": config.get("scheduler", {}).get("T_max", config["training"]["num_epochs"]),
            "step_size": config.get("scheduler", {}).get("step_size", 10),
            "gamma": config.get("scheduler", {}).get("gamma", 0.1),
            "eta_min": config.get("scheduler", {}).get("eta_min", 0.0),
            "patience": config.get("scheduler", {}).get("patience", 5),
            "mode": config.get("scheduler", {}).get("mode", "min"),
        },
        checkpoint_monitor=config["callbacks"].get("checkpoint_monitor", "val_accuracy"),
    )
    
    # Train
    print("\n" + "="*50)
    print("Starting training...")
    print("="*50)
    
    results = trainer.train()
    
    # Save results
    results_path = Path(config["callbacks"]["log_dir"]) / "training_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nTraining results saved to {results_path}")
    
    # Save class mapping
    mapping_path = Path(config["callbacks"]["checkpoint_dir"]) / "class_mapping.json"
    # Create reverse mapping (idx -> class_name)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    with open(mapping_path, "w") as f:
        json.dump(idx_to_class, f, indent=2)
    print(f"Class mapping saved to {mapping_path}")
    
    # Save model config
    config_path = Path(config["callbacks"]["checkpoint_dir"]) / "model_config.json"
    model_config = {
        "num_classes": len(class_to_idx),
        "model_type": config["model"]["type"],
        "input_size": config["model"]["input_size"],
        "class_to_idx": class_to_idx,
    }
    with open(config_path, "w") as f:
        json.dump(model_config, f, indent=2)
    print(f"Model config saved to {config_path}")
    
    print("\n" + "="*50)
    print("Training complete!")
    print("="*50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train crop disease detection model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training_config.yaml",
        help="Path to training config YAML file",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use for training",
    )
    
    args = parser.parse_args()
    main(args)
