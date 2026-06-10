#!/usr/bin/env python3
"""
Build dataset_metadata.json by scanning the actual dataset structure.
"""
from pathlib import Path
import json

dataset_root = Path('dataset')
split_counts = {}
class_set = set()

# Scan all splits
for split_name in ['train', 'val', 'test']:
    split_path = dataset_root / split_name
    if split_path.exists():
        for class_dir in sorted(split_path.iterdir()):
            if class_dir.is_dir():
                class_name = class_dir.name
                class_set.add(class_name)
                image_count = len(list(class_dir.glob('*')))
                
                if class_name not in split_counts:
                    split_counts[class_name] = {'train': 0, 'val': 0, 'test': 0}
                split_counts[class_name][split_name] = image_count

# Build metadata
total_images = sum(
    sum(counts.values()) 
    for counts in split_counts.values()
)

# Create class names and class_to_idx mappings
class_names = sorted(class_set)
class_to_idx = {cls: idx for idx, cls in enumerate(class_names)}

metadata = {
    'num_classes': len(class_set),
    'total_images': total_images,
    'class_names': class_names,
    'class_to_idx': class_to_idx,
    'split_counts': split_counts
}

# Save to file
output_path = dataset_root / 'dataset_metadata.json'
with open(output_path, 'w') as f:
    json.dump(metadata, f, indent=2)

# Print summary
print(f"✓ Metadata generated successfully")
print(f"  Classes: {metadata['num_classes']}")
print(f"  Total Images: {metadata['total_images']}")

train_total = sum(c['train'] for c in split_counts.values())
val_total = sum(c['val'] for c in split_counts.values())
test_total = sum(c['test'] for c in split_counts.values())

print(f"  Train: {train_total}")
print(f"  Val: {val_total}")
print(f"  Test: {test_total}")
print(f"\n  Saved to: {output_path}")

# Show first 3 classes
print(f"\nSample classes:")
for cls in sorted(class_set)[:3]:
    counts = split_counts[cls]
    print(f"  {cls}: train={counts['train']}, val={counts['val']}, test={counts['test']}")
