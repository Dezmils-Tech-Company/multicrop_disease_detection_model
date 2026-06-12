#!/usr/bin/env python
"""Quick dataset analysis using metadata."""
import json
from pathlib import Path
from collections import Counter


def analyze_dataset():
    """Analyze dataset from metadata."""
    metadata_path = "dataset/dataset_metadata.json"
    
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    
    print("\n" + "="*70)
    print("DATASET ANALYSIS REPORT")
    print("="*70)
    
    # Basic stats
    print(f"\n📊 BASIC STATISTICS:")
    print(f"  Total classes: {metadata['num_classes']}")
    print(f"  Total images: {metadata['total_images']:,}")
    
    # Split stats
    split_counts = {"train": 0, "val": 0, "test": 0}
    for class_name, splits in metadata["split_counts"].items():
        for split, count in splits.items():
            split_counts[split] += count
    
    print(f"\n📂 SPLIT DISTRIBUTION:")
    print(f"  Training: {split_counts['train']:,} ({100*split_counts['train']/metadata['total_images']:.1f}%)")
    print(f"  Validation: {split_counts['val']:,} ({100*split_counts['val']/metadata['total_images']:.1f}%)")
    print(f"  Test: {split_counts['test']:,} ({100*split_counts['test']/metadata['total_images']:.1f}%)")
    
    # Crop types
    crops = set()
    for class_name in metadata["class_names"]:
        crop = class_name.split("___")[0]
        crops.add(crop)
    
    print(f"\n🌾 CROP TYPES: {len(crops)}")
    for crop in sorted(crops):
        print(f"  - {crop}")
    
    # Class imbalance analysis
    train_counts = []
    class_names = []
    for class_name, splits in metadata["split_counts"].items():
        train_counts.append(splits["train"])
        class_names.append(class_name)
    
    min_train = min(train_counts)
    max_train = max(train_counts)
    avg_train = sum(train_counts) / len(train_counts)
    imbalance_ratio = max_train / min_train
    
    print(f"\n⚖️  CLASS IMBALANCE:")
    print(f"  Min samples/class: {min_train}")
    print(f"  Max samples/class: {max_train}")
    print(f"  Avg samples/class: {avg_train:.0f}")
    print(f"  Imbalance ratio: {imbalance_ratio:.2f}x")
    
    # Find extreme cases
    min_class = class_names[train_counts.index(min_train)]
    max_class = class_names[train_counts.index(max_train)]
    print(f"  Smallest class: {min_class} ({min_train} images)")
    print(f"  Largest class: {max_class} ({max_train} images)")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    print(f"  1. Use weighted Cross-Entropy Loss (imbalance ratio = {imbalance_ratio:.2f}x)")
    print(f"  2. Consider Focal Loss for very imbalanced classes")
    print(f"  3. Use batch normalization and dropout for regularization")
    print(f"  4. ResNet50 is suitable for this dataset size")
    print(f"  5. Start with learning rate 1e-4 for fine-tuning")
    
    print("\n" + "="*70 + "\n")
    
    return metadata


if __name__ == "__main__":
    analyze_dataset()
