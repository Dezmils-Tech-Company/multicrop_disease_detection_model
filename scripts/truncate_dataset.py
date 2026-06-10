"""
Truncate dataset classes that exceed 4000 images per class.
Maintains equal distribution across train/val/test partitions.
"""

import os
import shutil
import random
from pathlib import Path
from collections import defaultdict

# Set random seed for reproducibility
random.seed(42)

# Configuration
DATASET_ROOT = Path(__file__).parent.parent / "dataset"
MAX_IMAGES_PER_CLASS = 4000
PARTITIONS = ["train", "val", "test"]

def count_images_per_class():
    """Count images in each class across all partitions."""
    class_counts = defaultdict(lambda: {partition: 0 for partition in PARTITIONS})
    
    for partition in PARTITIONS:
        partition_path = DATASET_ROOT / partition
        if not partition_path.exists():
            continue
            
        for class_dir in partition_path.iterdir():
            if class_dir.is_dir():
                image_count = len(list(class_dir.glob("*.*")))
                class_counts[class_dir.name][partition] = image_count
    
    return class_counts

def truncate_class(class_name, partition_counts):
    """Truncate a class to MAX_IMAGES_PER_CLASS while maintaining partition ratios."""
    total_images = sum(partition_counts.values())
    
    if total_images <= MAX_IMAGES_PER_CLASS:
        print(f"  ✓ {class_name}: {total_images} images (no truncation needed)")
        return
    
    # Calculate target count per partition maintaining ratios
    reduction_ratio = MAX_IMAGES_PER_CLASS / total_images
    target_counts = {
        partition: max(1, int(partition_counts[partition] * reduction_ratio))
        for partition in PARTITIONS
    }
    
    # Adjust to ensure we hit exactly MAX_IMAGES_PER_CLASS
    current_total = sum(target_counts.values())
    if current_total < MAX_IMAGES_PER_CLASS:
        diff = MAX_IMAGES_PER_CLASS - current_total
        # Add to the partition with most images
        largest_partition = max(target_counts, key=lambda p: partition_counts[p])
        target_counts[largest_partition] += diff
    elif current_total > MAX_IMAGES_PER_CLASS:
        diff = current_total - MAX_IMAGES_PER_CLASS
        # Reduce from the partition with most excess
        for partition in sorted(PARTITIONS, key=lambda p: partition_counts[p], reverse=True):
            if diff <= 0:
                break
            reduction = min(diff, target_counts[partition] - 1)
            if reduction > 0:
                target_counts[partition] -= reduction
                diff -= reduction
    
    print(f"  Truncating {class_name}: {total_images} → {sum(target_counts.values())} images")
    
    # Truncate each partition
    for partition in PARTITIONS:
        partition_path = DATASET_ROOT / partition / class_name
        if not partition_path.exists():
            continue
        
        current_count = partition_counts[partition]
        target_count = target_counts[partition]
        
        if current_count <= target_count:
            print(f"    {partition}: {current_count} → {target_count} (no change)")
            continue
        
        # Get all images and randomly select which ones to delete
        all_images = list(partition_path.glob("*.*"))
        images_to_delete = random.sample(all_images, current_count - target_count)
        
        for image_path in images_to_delete:
            image_path.unlink()
        
        print(f"    {partition}: {current_count} → {target_count} (deleted {len(images_to_delete)})")

def main():
    print("=" * 70)
    print("DATASET TRUNCATION SCRIPT")
    print("=" * 70)
    print(f"Max images per class: {MAX_IMAGES_PER_CLASS}")
    print()
    
    # Count current state
    print("Counting current images per class...")
    class_counts = count_images_per_class()
    
    # Identify classes exceeding the limit
    classes_to_truncate = []
    for class_name, partition_counts in sorted(class_counts.items()):
        total = sum(partition_counts.values())
        if total > MAX_IMAGES_PER_CLASS:
            classes_to_truncate.append((class_name, partition_counts, total))
    
    if not classes_to_truncate:
        print("✓ No classes exceed the limit. Dataset is already balanced.")
        return
    
    print(f"\nFound {len(classes_to_truncate)} classes to truncate:\n")
    for class_name, _, total in classes_to_truncate:
        print(f"  • {class_name}: {total} images")
    
    # Confirm truncation
    print("\n" + "=" * 70)
    response = input("Proceed with truncation? (yes/no): ").strip().lower()
    if response != "yes":
        print("Truncation cancelled.")
        return
    
    print("\nTruncating classes...")
    print("-" * 70)
    
    total_deleted = 0
    for class_name, partition_counts, total in classes_to_truncate:
        truncate_class(class_name, partition_counts)
        total_deleted += total - MAX_IMAGES_PER_CLASS
    
    # Verify results
    print("\n" + "=" * 70)
    print("Verifying truncation...")
    updated_counts = count_images_per_class()
    
    max_in_class = 0
    for class_name, partition_counts in sorted(updated_counts.items()):
        total = sum(partition_counts.values())
        max_in_class = max(max_in_class, total)
    
    print(f"\nMax images per class after truncation: {max_in_class}")
    print(f"Total images deleted: {total_deleted}")
    print(f"\n✓ Truncation complete!")
    
    # Show summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    
    partition_totals = {partition: 0 for partition in PARTITIONS}
    for class_name, partition_counts in updated_counts.items():
        for partition in PARTITIONS:
            partition_totals[partition] += partition_counts[partition]
    
    total_images = sum(partition_totals.values())
    print(f"\nTotal images across dataset: {total_images}")
    for partition in PARTITIONS:
        pct = (partition_totals[partition] / total_images * 100) if total_images > 0 else 0
        print(f"  {partition}: {partition_totals[partition]} ({pct:.1f}%)")
    
    print(f"\nTotal classes: {len(updated_counts)}")

if __name__ == "__main__":
    main()
