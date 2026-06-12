#!/usr/bin/env python
"""Dataset audit script to validate and analyze the dataset."""
import json
import os
from pathlib import Path
from collections import defaultdict
from PIL import Image
import csv


def audit_dataset(dataset_path: str, output_file: str = "dataset_audit_report.json"):
    """
    Audit dataset for integrity and basic statistics.
    
    Args:
        dataset_path: Path to dataset root
        output_file: Path to save audit report
    """
    dataset_path = Path(dataset_path)
    report = {
        "timestamp": str(Path(__file__).parent),
        "dataset_path": str(dataset_path),
        "splits": {},
        "total_statistics": {},
        "issues": [],
    }
    
    # Load metadata
    metadata_file = dataset_path / "dataset_metadata.json"
    if metadata_file.exists():
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
        report["metadata"] = metadata
        expected_classes = metadata.get("num_classes")
        print(f"Metadata loaded: {expected_classes} classes")
    else:
        print("Warning: No metadata file found")
        expected_classes = None
    
    # Audit each split
    total_images = 0
    total_corrupted = 0
    all_classes = set()
    class_distribution = defaultdict(int)
    
    for split in ["train", "val", "test"]:
        split_path = dataset_path / split
        if not split_path.exists():
            report["issues"].append(f"Missing split: {split}")
            continue
        
        split_report = audit_split(split_path)
        report["splits"][split] = split_report
        
        total_images += split_report["total_images"]
        total_corrupted += split_report["corrupted_images"]
        all_classes.update(split_report["classes"].keys())
        
        for class_name, count in split_report["classes"].items():
            class_distribution[class_name] += count
        
        print(f"\n{split.upper()} split:")
        print(f"  Total images: {split_report['total_images']}")
        print(f"  Valid images: {split_report['valid_images']}")
        print(f"  Corrupted: {split_report['corrupted_images']}")
        print(f"  Classes: {len(split_report['classes'])}")
    
    # Total statistics
    report["total_statistics"] = {
        "total_images": total_images,
        "valid_images": total_images - total_corrupted,
        "corrupted_images": total_corrupted,
        "unique_classes": len(all_classes),
        "expected_classes": expected_classes,
        "class_distribution": dict(sorted(class_distribution.items())),
    }
    
    # Class imbalance analysis
    if class_distribution:
        counts = list(class_distribution.values())
        min_count = min(counts)
        max_count = max(counts)
        avg_count = sum(counts) / len(counts)
        
        imbalance_ratio = max_count / min_count if min_count > 0 else 0
        report["class_imbalance"] = {
            "min_samples": min_count,
            "max_samples": max_count,
            "avg_samples": round(avg_count, 1),
            "imbalance_ratio": round(imbalance_ratio, 2),
        }
        
        print(f"\nClass Imbalance:")
        print(f"  Min: {min_count}, Max: {max_count}, Avg: {avg_count:.1f}")
        print(f"  Imbalance ratio: {imbalance_ratio:.2f}x")
    
    # Check for issues
    if expected_classes and len(all_classes) != expected_classes:
        report["issues"].append(
            f"Class count mismatch: found {len(all_classes)}, expected {expected_classes}"
        )
    
    if total_corrupted > 0:
        report["issues"].append(f"Found {total_corrupted} corrupted images")
    
    # Save report
    with open(output_file, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\nAudit report saved to {output_file}")
    return report


def audit_split(split_path: Path) -> dict:
    """Audit a single data split."""
    split_report = {
        "total_images": 0,
        "valid_images": 0,
        "corrupted_images": 0,
        "classes": defaultdict(int),
        "corrupted_files": [],
    }
    
    for class_dir in split_path.iterdir():
        if not class_dir.is_dir():
            continue
        
        class_name = class_dir.name
        
        for img_file in class_dir.iterdir():
            if img_file.suffix.lower() in (".jpg", ".jpeg", ".png"):
                split_report["total_images"] += 1
                
                # Check if image is valid
                try:
                    with Image.open(img_file) as img:
                        img.verify()
                    split_report["valid_images"] += 1
                    split_report["classes"][class_name] += 1
                except Exception as e:
                    split_report["corrupted_images"] += 1
                    split_report["corrupted_files"].append(str(img_file))
    
    # Convert defaultdict to regular dict
    split_report["classes"] = dict(split_report["classes"])
    return split_report


if __name__ == "__main__":
    import sys
    
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else "dataset"
    
    print(f"Auditing dataset at: {dataset_path}")
    print("=" * 60)
    
    report = audit_dataset(dataset_path)
    
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print(f"  Total images: {report['total_statistics']['total_images']:,}")
    print(f"  Valid images: {report['total_statistics']['valid_images']:,}")
    print(f"  Corrupted: {report['total_statistics']['corrupted_images']}")
    print(f"  Unique classes: {report['total_statistics']['unique_classes']}")
    
    if report["issues"]:
        print("\nISSUES FOUND:")
        for issue in report["issues"]:
            print(f"  ⚠ {issue}")
    else:
        print("\n✓ No issues found!")
