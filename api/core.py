import json
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import yaml

from inference import Predictor

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_FILE = BASE_DIR / "configs" / "inference_config.yaml"


def load_api_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    config_file = Path(config_path) if config_path else DEFAULT_CONFIG_FILE
    if not config_file.exists():
        raise FileNotFoundError(f"API config not found at {config_file}")

    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config or {}


def load_class_mapping_from_config(config: Dict[str, Any]) -> Optional[Dict[int, str]]:
    model_metadata = config.get("model_metadata", {})
    class_to_idx = model_metadata.get("class_to_idx") or config.get("class_to_idx")

    if class_to_idx:
        return {int(v): k for k, v in class_to_idx.items()}

    return None


_predictor = None


def create_predictor(config_path: Optional[str] = None) -> Predictor:
    config = load_api_config(config_path)
    inference_cfg = config.get("inference", {})

    model_checkpoint = inference_cfg.get("model_checkpoint", "checkpoints/best_model.pth")
    model_checkpoint_path = (BASE_DIR / model_checkpoint).resolve()
    device = inference_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    model_config_path = None
    if model_checkpoint_path.parent.exists():
        candidate = model_checkpoint_path.parent / "model_config.json"
        if candidate.exists():
            model_config_path = str(candidate)

    predictor = Predictor(
        model_path=str(model_checkpoint_path),
        config_path=model_config_path,
        device=device,
    )
    return predictor


def get_predictor(config_path: Optional[str] = None) -> Predictor:
    global _predictor
    if _predictor is None:
        _predictor = create_predictor(config_path)
    return _predictor
