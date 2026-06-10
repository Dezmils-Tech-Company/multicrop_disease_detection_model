#!/usr/bin/env python3
"""Fast dataset truncation with batch operations."""

import os
import random
from pathlib import Path
from collections import defaultdict
import shutil

MAX_IMAGES_PER_CLASS = 4000
DATASET_ROOT = Path(__file__).parent.parent / "dataset"
PARTITIONS = ["train", "val", "test"]

random.seed(42)

def count_images():
    """Count images per class and partition."""
    counts = defaultdict(lambda: {p: 0 for p in PARTITIONS})
    
    for partition in PARTITIONS:
        part_path = DATASET_ROOT / partition
        if not part_path.exists():
            continue
        for class_dir in part_path.iterdir():
            if class_dir.is_dir():
                img_count = len(list(class_dir.glob("*.*")))
                counts[class_dir.name][partition] = img_count
    
    return counts

def truncate_class(class_name, counts):
    """Truncate a single class."""
    total = sum(counts.values())
    if total <= MAX_IMAGES_PER_CLASS:
        print(f"  OK {class_name}: {total} images")
        return
    
    # Calculate target distribution
    ratio = MAX_IMAGES_PER_CLASS / total
    targets = {p: max(1, int(counts[p] * ratio)) for p in PARTITIONS}
    
    # Adjust to exact total
    current = sum(targets.values())
    if current < MAX_IMAGES_PER_CLASS:
        diff = MAX_IMAGES_PER_CLASS - current
        largest = max(PARTITIONS, key=lambda p: counts[p])
        targets[largest] += diff
    elif current > MAX_IMAGES_PER_CLASS:
        diff = current - MAX_IMAGES_PER_CLASS
        for p in sorted(PARTITIONS, key=lambda x: counts[x], reverse=True):
            if diff <= 0:
                break
            reduction = min(diff, targets[p] - 1)
            if reduction > 0:
                targets[p] -= reduction
                diff -= reduction
    
    print(f"  Truncating {class_name}: {total} -> {sum(targets.values())}")
    
    # Delete files from each partition
    for partition in PARTITIONS:
        class_path = DATASET_ROOT / partition / class_name
        if not class_path.exists():
            continue
        
        current_cnt = counts[partition]
        target_cnt = targets[partition]
        
        if current_cnt <= target_cnt:
            print(f"    {partition}: {current_cnt} -> {target_cnt}")
            continue
        
        # Get all images and delete excess
        all_imgs = list(class_path.glob("*.*"))
        to_delete = random.sample(all_imgs, current_cnt - target_cnt)
        
        for img_path in to_delete:
            try:
                img_path.unlink()
            except:
                pass
        
        print(f"    {partition}: {current_cnt} -> {target_cnt} (deleted {len(to_delete)})")

# Main
print("=" * 70)
print("FAST DATASET TRUNCATION")
print("=" * 70)
print(f"Max per class: {MAX_IMAGES_PER_CLASS}")
print()

print("Counting images...")
all_counts = count_images()

# Find classes to truncate
to_truncate = []
for class_name, counts in sorted(all_counts.items()):
    total = sum(counts.values())
    if total > MAX_IMAGES_PER_CLASS:
        to_truncate.append((class_name, counts, total))

if not to_truncate:
    print("OK - No truncation needed")
    exit(0)

print(f"Found {len(to_truncate)} classes to truncate:")
print()
for name, _, total in to_truncate:
    print(f"  * {name}: {total} images")

print()
print("=" * 70)
resp = input("Proceed? (yes/no): ").strip().lower()
if resp != "yes":
    print("Cancelled")
    exit(0)

print()
print("Truncating...")
print("-" * 70)

total_deleted = 0
for name, counts, orig_total in to_truncate:
    truncate_class(name, counts)
    total_deleted += orig_total - MAX_IMAGES_PER_CLASS

# Summary
print()
print("=" * 70)
print("VERIFICATION")
print("=" * 70)

updated = count_images()
max_per_class = max((sum(c.values()) for c in updated.values()), default=0)

print(f"Max per class: {max_per_class}")
print(f"Total deleted: {total_deleted}")
print("OK - Done!")

# Stats
print()
print("=" * 70)
print("STATISTICS")
print("=" * 70)
print()

totals = {p: 0 for p in PARTITIONS}
for counts in updated.values():
    for p in PARTITIONS:
        totals[p] += counts[p]

grand_total = sum(totals.values())
print(f"Total images: {grand_total}")
for p in PARTITIONS:
    pct = (totals[p] / grand_total * 100) if grand_total > 0 else 0
    print(f"  {p}: {totals[p]} ({pct:.1f}%)")

print()
print(f"Total classes: {len(updated)}")
