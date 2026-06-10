from fastapi import APIRouter
from ..schemas import HealthResponse

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/", response_model=HealthResponse)
def health_check():
    return {"status": "ok", "message": "Crop Disease Detection API is healthy."}
