from typing import Dict, Optional

from .disease_detector import DiseaseDetectionModel


def create_model_from_config(config: Dict[str, Optional[object]], **override_kwargs) -> DiseaseDetectionModel:
    """Create a DiseaseDetectionModel from a configuration dictionary."""
    model_args = {
        "num_classes": int(config.get("num_classes", 53)),
        "model_type": config.get("model_type", "resnet50"),
        "pretrained": config.get("pretrained", True),
        "freeze_backbone": int(config.get("freeze_backbone", 0)),
        "head_hidden_dims": config.get("head_hidden_dims", [512]),
        "dropout_rate": float(config.get("dropout_rate", 0.3)),
    }
    model_args.update(override_kwargs)
    return DiseaseDetectionModel(**model_args)
