from fastapi import APIRouter, HTTPException, UploadFile, File
from typing import List, Optional
from pathlib import Path
import tempfile
import shutil

from ..schemas import PredictionResult
from ..core import get_predictor

router = APIRouter(prefix="/predict", tags=["prediction"])


def _save_upload(file: UploadFile) -> str:
    temp_dir = tempfile.mkdtemp(prefix="crop_predict_")
    temp_path = Path(temp_dir) / file.filename
    try:
        with temp_path.open("wb") as f:
            shutil.copyfileobj(file.file, f)
    finally:
        file.file.close()
    return str(temp_path)


@router.post("/single", response_model=PredictionResult)
def predict_single(
    image: UploadFile = File(...),
    top_k: Optional[int] = 3,
    confidence_threshold: Optional[float] = 0.0,
):
    temp_path = _save_upload(image)
    try:
        predictor = get_predictor()
        result = predictor.predict_image(temp_path, top_k=top_k, confidence_threshold=confidence_threshold)
        if not result["predictions"]:
            raise HTTPException(status_code=400, detail="No valid predictions returned.")
        return result
    finally:
        try:
            Path(temp_path).unlink()
            Path(temp_path).parent.rmdir()
        except Exception:
            pass


@router.post("/batch", response_model=List[PredictionResult])
def predict_batch(
    images: List[UploadFile] = File(...),
    top_k: Optional[int] = 3,
    confidence_threshold: Optional[float] = 0.0,
):
    temp_paths = []
    results = []

    try:
        for upload in images:
            temp_paths.append(_save_upload(upload))

        predictor = get_predictor()
        results = predictor.predict_batch(temp_paths, top_k=top_k)
        if confidence_threshold > 0.0:
            for result in results:
                result["predictions"] = [p for p in result["predictions"] if p["confidence"] >= confidence_threshold]
                result["top_prediction"] = result["predictions"][0] if result["predictions"] else None

        return results
    finally:
        for path in temp_paths:
            try:
                Path(path).unlink()
                Path(path).parent.rmdir()
            except Exception:
                pass
