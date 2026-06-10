"""PyTorch Dataset class for crop disease detection."""
import os
from pathlib import Path
from typing import Optional, Tuple, Dict, List
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class CropDiseaseDataset(Dataset):
    """
    Custom PyTorch Dataset for crop disease images.
    
    Loads images from directory structure where each subdirectory
    is a class label (e.g., "Tomato___Leaf_Spot").
    """
    
    def __init__(
        self,
        root_dir: str,
        transform=None,
        class_to_idx: Optional[Dict[str, int]] = None,
        allowed_extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png"),
    ):
        """
        Args:
            root_dir: Path to dataset root directory with class subdirectories
            transform: Optional torchvision transforms to apply
            class_to_idx: Optional pre-defined class to index mapping
            allowed_extensions: Tuple of allowed image extensions
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.allowed_extensions = tuple(ext.lower() for ext in allowed_extensions)
        
        # Build class to index mapping
        if class_to_idx is None:
            self.classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
            self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        else:
            self.class_to_idx = class_to_idx
            self.classes = sorted(class_to_idx.keys())
        
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        
        # Build image paths and labels
        self.samples = self._load_samples()
        
    def _load_samples(self) -> List[Tuple[Path, int]]:
        """Load all valid image paths and their class indices."""
        samples = []
        errors = []
        
        for class_name, class_idx in self.class_to_idx.items():
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                continue
            
            for img_file in class_dir.iterdir():
                if img_file.suffix.lower() in self.allowed_extensions:
                    samples.append((img_file, class_idx))
                elif img_file.is_file():
                    errors.append(str(img_file))
        
        if errors:
            print(f"Warning: {len(errors)} files skipped (invalid extensions)")
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        """
        Returns:
            Tuple of (image_tensor, class_index, class_name)
        """
        img_path, class_idx = self.samples[idx]
        class_name = self.idx_to_class[class_idx]
        
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return black image as fallback
            image = Image.new("RGB", (224, 224), color=(0, 0, 0))
        
        if self.transform:
            image = self.transform(image)
        else:
            image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        
        return image, class_idx, class_name
    
    def get_class_distribution(self) -> Dict[str, int]:
        """Return count of images per class."""
        dist = {}
        for _, class_idx in self.samples:
            class_name = self.idx_to_class[class_idx]
            dist[class_name] = dist.get(class_name, 0) + 1
        return dist
    
    def get_class_weights(self) -> torch.Tensor:
        """
        Compute class weights for imbalanced data.
        Returns: Weight tensor (inverse of class frequency)
        """
        dist = self.get_class_distribution()
        total = len(self.samples)
        weights = torch.zeros(len(self.classes))
        
        for class_idx, class_name in self.idx_to_class.items():
            class_count = dist.get(class_name, 1)
            weights[class_idx] = total / (len(self.classes) * class_count)
        
        return weights
