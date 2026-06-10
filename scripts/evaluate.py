#!/usr/bin/env python
"""Evaluation script for test set evaluation."""
import argparse
import json
import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from data import create_dataloaders, CropDiseaseDataset
from models import DiseaseDetectionModel
from evaluation import Evaluator


def evaluate_model(model_path: str, device: str = "cuda"):
    """Evaluate model on test set."""
    print("\n" + "="*70)
    print("MODEL EVALUATION")
    print("="*70)
    
    # Load model config
    checkpoint_dir = Path(model_path).parent
    config_path = checkpoint_dir / "model_config.json"
    
    if not config_path.exists():
        print(f"Error: Model config not found at {config_path}")
        return None
    
    with open(config_path, "r") as f:
        config = json.load(f)
    
    print(f"\nModel configuration:")
    print(f"  Classes: {config['num_classes']}")
    print(f"  Model type: {config['model_type']}")
    print(f"  Input size: {config['input_size']}")
    
    # Create model
    print(f"\nLoading model from {model_path}...")
    model = DiseaseDetectionModel(
        num_classes=config['num_classes'],
        model_type=config['model_type'],
        pretrained=False,
    )
    
    # Load weights
    checkpoint = torch.load(model_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    
    # Create test dataloader
    print("\nCreating test dataloader...")
    _, _, test_loader, class_to_idx = create_dataloaders(
        train_dir="dataset/train",
        val_dir="dataset/val",
        test_dir="dataset/test",
        batch_size=64,
        num_workers=4,
        image_size=config['input_size'],
    )
    
    # Create reverse mapping
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    
    # Evaluate
    print("\nEvaluating on test set...")
    evaluator = Evaluator(
        model=model,
        test_loader=test_loader,
        class_mapping=idx_to_class,
        device=device,
    )
    
    results = evaluator.evaluate()
    
    # Print results
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    
    metrics = results['overall_metrics']
    print(f"\nOverall Metrics:")
    print(f"  Loss: {metrics.get('loss', 'N/A'):.4f}")
    print(f"  Accuracy: {metrics.get('accuracy', 'N/A'):.2f}%")
    print(f"  Top-5 Accuracy: {metrics.get('top5_accuracy', 'N/A'):.2f}%")
    print(f"  Macro Accuracy: {metrics.get('macro_accuracy', 'N/A'):.2f}%")
    
    # Top performing classes
    per_class = results['per_class_metrics']
    if per_class:
        print(f"\nBest performing classes:")
        sorted_classes = sorted(per_class.items(), key=lambda x: x[1]['accuracy'], reverse=True)[:5]
        for idx, metrics in sorted_classes:
            class_name = idx_to_class.get(int(idx), f"class_{idx}")
            print(f"  {class_name:45} {metrics['accuracy']:6.2f}% ({metrics['count']} samples)")
        
        print(f"\nWorst performing classes:")
        sorted_classes = sorted(per_class.items(), key=lambda x: x[1]['accuracy'])[:5]
        for idx, metrics in sorted_classes:
            class_name = idx_to_class.get(int(idx), f"class_{idx}")
            print(f"  {class_name:45} {metrics['accuracy']:6.2f}% ({metrics['count']} samples)")
    
    # Save results
    output_dir = Path("logs")
    evaluator.save_results(results, output_dir=str(output_dir))
    
    print(f"\n[OK] Evaluation completed and results saved to {output_dir}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained model on test set")
    parser.add_argument("--model", required=True, help="Path to trained model checkpoint")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    
    args = parser.parse_args()
    
    if not Path(args.model).exists():
        print(f"Error: Model file not found at {args.model}")
        sys.exit(1)
    
    evaluate_model(args.model, device=args.device)


if __name__ == "__main__":
    main()
