from typing import Dict, Tuple


def parse_class_name(class_name: str) -> Tuple[str, str]:
    """Parse crop and disease from a class name string."""
    if "___" in class_name:
        crop, disease = class_name.split("___", 1)
    else:
        crop = class_name
        disease = "Unknown"
    return crop, disease


def format_prediction(class_index: int, class_name: str, confidence: float) -> Dict[str, object]:
    crop, disease = parse_class_name(class_name)
    return {
        "class_index": class_index,
        "class_name": class_name,
        "crop": crop,
        "disease": disease,
        "confidence": float(confidence),
    }
