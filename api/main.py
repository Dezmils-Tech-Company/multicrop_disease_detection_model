from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import uvicorn

from .endpoints import router as api_router
from .core import load_api_config

app = FastAPI(
    title="Crop Disease Detection API",
    description="API for crop type and disease classification.",
    version="1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)

config = load_api_config()
api_config = config.get("api", {})


if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host=api_config.get("host", "0.0.0.0"),
        port=api_config.get("port", 8000),
        reload=api_config.get("reload", False),
        workers=api_config.get("workers", 1),
    )
