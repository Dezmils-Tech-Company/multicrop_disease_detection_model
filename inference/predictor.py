"""Model predictor for inference."""
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Tuple, Dict, List, Optional
import json
from PIL import Image
import numpy as np

from models import DiseaseDetectionModel
from data import ImagePreprocessor, get_inference_transforms


class Predictor:
    """Load model and make predictions."""
    
    def __init__(
        self,
        model_path: str,
        config_path: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Args:
            model_path: Path to saved model checkpoint
            config_path: Path to model config JSON
            device: Device for inference
        """
        self.device = device
        self.model_path = Path(model_path)
        
        # Load model config
        if config_path:
            with open(config_path, "r") as f:
                self.config = json.load(f)
        else:
            self.config = self._load_config_from_checkpoint()
        
        # Load model
        self.model = self._load_model()
        self.model.to(device)
        self.model.eval()
        
        # Preprocessing
        self.preprocessor = ImagePreprocessor(target_size=(224, 224))
        self.transforms = get_inference_transforms(image_size=224)
        
        # Load class mapping
        self.class_mapping = None
        if config_path:
            config_dir = Path(config_path).parent
            mapping_file = config_dir / "class_mapping.json"
            if mapping_file.exists():
                with open(mapping_file, "r") as f:
                    self.class_mapping = json.load(f)
    
    def _load_config_from_checkpoint(self) -> Dict:
        """Load config from checkpoint file."""
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            return checkpoint.get("config", {})
        except:
            return {}
    
    def _load_model(self) -> DiseaseDetectionModel:
        """Load model from checkpoint."""
        # Create model
        model = DiseaseDetectionModel(
            num_classes=self.config.get("num_classes", 53),
            model_type=self.config.get("model_type", "resnet50"),
            pretrained=False,  # Already loaded weights
        )
        
        # Load weights
        checkpoint = torch.load(self.model_path, map_location=self.device)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        
        return model
    
    def predict_image(
        self,
        image_path: str,
        top_k: int = 3,
        confidence_threshold: float = 0.0,
    ) -> Dict:
        """
        Predict on single image.
        
        Args:
            image_path: Path to image file
            top_k: Return top-k predictions
            confidence_threshold: Only return predictions above this threshold
        
        Returns:
            Dict with predictions
        """
        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transforms(image).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)[0]
        
        # Get top predictions
        top_probs, top_indices = torch.topk(probabilities, k=min(top_k, len(probabilities)))
        
        predictions = []
        for prob, idx in zip(top_probs, top_indices):
            prob_val = prob.item()
            if prob_val >= confidence_threshold:
                class_idx = idx.item()
                class_name = self.model.idx_to_class[class_idx] if hasattr(self.model, 'idx_to_class') else f"class_{class_idx}"
                
                # Parse crop and disease from class name
                crop, disease = self._parse_class_name(class_name)
                
                predictions.append({
                    "class_index": class_idx,
                    "class_name": class_name,
                    "crop": crop,
                    "disease": disease,
                    "confidence": float(prob_val),
                })
        
        return {
            "image_path": str(image_path),
            "predictions": predictions,
            "top_prediction": predictions[0] if predictions else None,
        }
    
    def predict_batch(
        self,
        image_paths: List[str],
        top_k: int = 3,
        batch_size: int = 32,
    ) -> List[Dict]:
        """
        Predict on multiple images.
        
        Args:
            image_paths: List of image file paths
            top_k: Return top-k predictions per image
            batch_size: Batch size for inference
        
        Returns:
            List of prediction results
        """
        results = []
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            
            # Load and preprocess batch
            batch_images = []
            valid_paths = []
            
            for path in batch_paths:
                try:
                    image = Image.open(path).convert("RGB")
                    image_tensor = self.transforms(image)
                    batch_images.append(image_tensor)
                    valid_paths.append(path)
                except Exception as e:
                    print(f"Error loading image {path}: {e}")
            
            if not batch_images:
                continue
            
            batch_tensor = torch.stack(batch_images).to(self.device)
            
            # Predict
            with torch.no_grad():
                outputs = self.model(batch_tensor)
                probabilities = F.softmax(outputs, dim=1)
            
            # Process results
            for j, path in enumerate(valid_paths):
                top_probs, top_indices = torch.topk(probabilities[j], k=min(top_k, probabilities.size(1)))
                
                predictions = []
                for prob, idx in zip(top_probs, top_indices):
                    class_idx = idx.item()
                    class_name = f"class_{class_idx}"
                    crop, disease = self._parse_class_name(class_name)
                    
                    predictions.append({
                        "class_index": class_idx,
                        "class_name": class_name,
                        "crop": crop,
                        "disease": disease,
                        "confidence": float(prob.item()),
                    })
                
                results.append({
                    "image_path": str(path),
                    "predictions": predictions,
                    "top_prediction": predictions[0] if predictions else None,
                })
        
        return results
    
    @staticmethod
    def _parse_class_name(class_name: str) -> Tuple[str, str]:
        """
        Parse crop and disease from class name.
        Expected format: "Crop___Disease"
        """
        if "___" in class_name:
            parts = class_name.split("___")
            crop = parts[0]
            disease = parts[1]
        else:
            crop = class_name
            disease = "Unknown"
        
        return crop, disease
