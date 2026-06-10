#!/usr/bin/env python
"""Prediction script for single or batch image inference."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from inference import Predictor


def predict_single(model_path: str, image_path: str, top_k: int = 3):
    """Predict on a single image."""
    print(f"\nLoading model from {model_path}...")
    predictor = Predictor(model_path, device="cuda")
    
    print(f"Predicting on image: {image_path}...")
    result = predictor.predict_image(image_path, top_k=top_k)
    
    print("\n" + "="*70)
    print("PREDICTION RESULTS")
    print("="*70)
    print(f"Image: {result['image_path']}")
    
    if result['top_prediction']:
        print(f"\nTop Prediction:")
        top = result['top_prediction']
        print(f"  Class: {top['class_name']}")
        print(f"  Crop: {top['crop']}")
        print(f"  Disease/Status: {top['disease']}")
        print(f"  Confidence: {top['confidence']:.2%}")
    
    print(f"\nTop {top_k} Predictions:")
    for i, pred in enumerate(result['predictions'][:top_k], 1):
        print(f"  {i}. {pred['class_name']:45} ({pred['confidence']:.2%})")
    
    return result


def predict_batch(model_path: str, image_dir: str, top_k: int = 3):
    """Predict on multiple images in a directory."""
    image_dir = Path(image_dir)
    image_paths = list(image_dir.glob("**/*.jpg")) + list(image_dir.glob("**/*.png"))
    
    if not image_paths:
        print(f"No images found in {image_dir}")
        return []
    
    print(f"\nFound {len(image_paths)} images")
    print(f"Loading model from {model_path}...")
    predictor = Predictor(model_path, device="cuda")
    
    print(f"Predicting on {len(image_paths)} images...")
    results = predictor.predict_batch(image_paths, top_k=top_k, batch_size=32)
    
    print("\n" + "="*70)
    print("BATCH PREDICTION RESULTS")
    print("="*70)
    print(f"Total images: {len(results)}")
    
    for i, result in enumerate(results[:10], 1):  # Show first 10
        top = result['top_prediction']
        if top:
            print(f"{i:3}. {Path(result['image_path']).name:40} -> {top['crop']:12} / {top['disease']:25} ({top['confidence']:.2%})")
    
    if len(results) > 10:
        print(f"... and {len(results) - 10} more")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Make predictions with trained model")
    parser.add_argument("--model", required=True, help="Path to trained model checkpoint")
    parser.add_argument("--image", help="Path to single image for prediction")
    parser.add_argument("--batch", help="Path to directory of images for batch prediction")
    parser.add_argument("--top-k", type=int, default=3, help="Return top-k predictions")
    parser.add_argument("--output", help="Save results to JSON file")
    
    args = parser.parse_args()
    
    results = None
    
    if args.image:
        results = predict_single(args.model, args.image, args.top_k)
    elif args.batch:
        results = predict_batch(args.model, args.batch, args.top_k)
    else:
        print("Error: Provide either --image or --batch")
        sys.exit(1)
    
    # Save results if requested
    if args.output and results:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
