"""Model predictor for inference."""
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image

from models import DiseaseDetectionModel
from data import get_inference_transforms
from .postprocessor import format_prediction


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
        self.config_path = Path(config_path) if config_path else None

        # Load config and class mapping
        self.config = self._load_config(config_path)
        self.idx_to_class = self._load_class_mapping()

        # Load model
        self.model = self._load_model()
        self.model.to(self.device)
        self.model.eval()

        # Preprocessing
        image_size = int(self.config.get("input_size", 224))
        self.transforms = get_inference_transforms(image_size=image_size)

    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load model configuration from config path or checkpoint."""
        if config_path:
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass

        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            return checkpoint.get("config", {})
        except Exception:
            return {}

    def _load_class_mapping(self) -> Dict[int, str]:
        """Load class index to name mapping."""
        mapping = {}

        # If config contains a class mapping or class_to_idx mapping, use it
        if self.config_path:
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    config_data = json.load(f)
                if "class_to_idx" in config_data:
                    mapping = {int(v): k for k, v in config_data["class_to_idx"].items()}
                elif "class_mapping" in config_data:
                    mapping = {int(k): v for k, v in config_data["class_mapping"].items()}
            except Exception:
                pass

        if not mapping:
            config_dir = self.config_path.parent if self.config_path else self.model_path.parent
            mapping_file = config_dir / "class_mapping.json"
            if mapping_file.exists():
                try:
                    with open(mapping_file, "r", encoding="utf-8") as f:
                        mapping = {int(k): v for k, v in json.load(f).items()}
                except Exception:
                    mapping = {}

        return mapping

    def _load_model(self) -> DiseaseDetectionModel:
        """Load model from checkpoint."""
        model = DiseaseDetectionModel(
            num_classes=int(self.config.get("num_classes", 53)),
            model_type=self.config.get("model_type", "resnet50"),
            pretrained=False,
        )

        checkpoint = torch.load(self.model_path, map_location=self.device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)
        return model

    def _get_class_name(self, class_idx: int) -> str:
        return self.idx_to_class.get(class_idx, f"class_{class_idx}")

    def _predict_tensor(
        self, tensor: torch.Tensor, top_k: int, confidence_threshold: float
    ) -> Dict[str, object]:
        with torch.no_grad():
            outputs = self.model(tensor)
            probabilities = F.softmax(outputs, dim=1)

        top_probs, top_indices = torch.topk(probabilities, k=min(top_k, probabilities.size(1)), dim=1)
        predictions = []

        for prob, idx in zip(top_probs[0], top_indices[0]):
            prob_val = float(prob.item())
            if prob_val < confidence_threshold:
                continue
            class_idx = int(idx.item())
            class_name = self._get_class_name(class_idx)
            predictions.append(format_prediction(class_idx, class_name, prob_val))

        return {
            "image_path": "",
            "predictions": predictions,
            "top_prediction": predictions[0] if predictions else None,
        }

    def predict_image(
        self,
        image_path: str,
        top_k: int = 3,
        confidence_threshold: float = 0.0,
    ) -> Dict:
        """Predict on a single image file path."""
        image = Image.open(image_path).convert("RGB")
        tensor = self.transforms(image).unsqueeze(0).to(self.device)
        result = self._predict_tensor(tensor, top_k, confidence_threshold)
        result["image_path"] = str(image_path)
        return result

    def predict_pil_image(
        self,
        image: Image.Image,
        top_k: int = 3,
        confidence_threshold: float = 0.0,
    ) -> Dict:
        """Predict on a PIL image object."""
        tensor = self.transforms(image.convert("RGB")).unsqueeze(0).to(self.device)
        result = self._predict_tensor(tensor, top_k, confidence_threshold)
        return result

    def predict_batch(
        self,
        image_paths: List[str],
        top_k: int = 3,
        batch_size: int = 32,
    ) -> List[Dict]:
        results = []
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i : i + batch_size]
            batch_images = []
            valid_paths = []

            for path in batch_paths:
                try:
                    image = Image.open(path).convert("RGB")
                    batch_images.append(self.transforms(image))
                    valid_paths.append(path)
                except Exception as e:
                    print(f"Error loading image {path}: {e}")

            if not batch_images:
                continue

            batch_tensor = torch.stack(batch_images).to(self.device)
            with torch.no_grad():
                outputs = self.model(batch_tensor)
                probabilities = F.softmax(outputs, dim=1)

            for j, path in enumerate(valid_paths):
                top_probs, top_indices = torch.topk(
                    probabilities[j], k=min(top_k, probabilities.size(1)), dim=0
                )
                predictions = []
                for prob, idx in zip(top_probs, top_indices):
                    class_idx = int(idx.item())
                    class_name = self._get_class_name(class_idx)
                    predictions.append(format_prediction(class_idx, class_name, float(prob.item())))

                results.append(
                    {
                        "image_path": str(path),
                        "predictions": predictions,
                        "top_prediction": predictions[0] if predictions else None,
                    }
                )

        return results
