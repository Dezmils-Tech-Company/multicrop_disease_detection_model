from pydantic import BaseModel
from typing import List, Optional


class PredictionItem(BaseModel):
    class_index: int
    class_name: str
    crop: str
    disease: str
    confidence: float


class PredictionResult(BaseModel):
    image_path: str
    predictions: List[PredictionItem]
    top_prediction: Optional[PredictionItem] = None


class HealthResponse(BaseModel):
    status: str
    message: str
