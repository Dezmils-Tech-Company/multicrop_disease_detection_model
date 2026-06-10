"""Image preprocessing utilities."""
from PIL import Image
import numpy as np
from pathlib import Path
from typing import Tuple, Optional


class ImagePreprocessor:
    """Utility class for image preprocessing and validation."""
    
    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        """
        Args:
            target_size: Target (height, width) for resizing
        """
        self.target_size = target_size
    
    def load_image(self, image_path: str) -> Optional[Image.Image]:
        """
        Load image with error handling.
        
        Returns:
            PIL Image or None if loading fails
        """
        try:
            img = Image.open(image_path).convert("RGB")
            return img
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None
    
    def preprocess(self, image_path: str) -> Optional[np.ndarray]:
        """
        Load, resize, and normalize image to numpy array.
        
        Returns:
            Normalized numpy array (H, W, C) or None on error
        """
        img = self.load_image(image_path)
        if img is None:
            return None
        
        img = img.resize(self.target_size, Image.BILINEAR)
        arr = np.array(img).astype(np.float32) / 255.0
        return arr
    
    def validate_image(self, image_path: str) -> bool:
        """Check if image file is valid and loadable."""
        try:
            img = Image.open(image_path)
            img.verify()
            return True
        except Exception:
            return False
    
    @staticmethod
    def check_dataset_integrity(root_dir: str, sample_size: Optional[int] = None) -> dict:
        """
        Check dataset for corrupted or invalid images.
        
        Args:
            root_dir: Root directory of dataset
            sample_size: Number of images to check per class (None = all)
        
        Returns:
            Dict with statistics about corrupted images
        """
        root_path = Path(root_dir)
        preprocessor = ImagePreprocessor()
        
        stats = {
            "total_images": 0,
            "valid_images": 0,
            "corrupted_images": [],
            "classes": {}
        }
        
        for class_dir in root_path.iterdir():
            if not class_dir.is_dir():
                continue
            
            class_name = class_dir.name
            stats["classes"][class_name] = {"valid": 0, "corrupted": 0}
            
            img_files = list(class_dir.glob("*"))
            if sample_size:
                img_files = img_files[:sample_size]
            
            for img_file in img_files:
                if img_file.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    stats["total_images"] += 1
                    if preprocessor.validate_image(str(img_file)):
                        stats["valid_images"] += 1
                        stats["classes"][class_name]["valid"] += 1
                    else:
                        stats["corrupted_images"].append(str(img_file))
                        stats["classes"][class_name]["corrupted"] += 1
        
        return stats
