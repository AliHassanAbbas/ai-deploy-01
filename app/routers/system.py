"""Public endpoints: welcome and health. No API key required."""
from fastapi import APIRouter, Depends

from app.config import settings
from app.deps import get_predictor
from app.schemas import HealthOut

router = APIRouter(tags=["system"])


@router.get("/")
def read_root():
    return {"message": settings.app_name, "docs": "http://127.0.0.1:8000/docs"}


@router.get("/health", response_model=HealthOut)
def health(predictor=Depends(get_predictor)):
    return {"status": "ok", "model_version": predictor.config["model_version"]}