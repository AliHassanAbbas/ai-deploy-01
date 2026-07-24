"""All request/response shapes (Pydantic models) in one place."""
from pydantic import BaseModel, Field


# ---- Requests ----------------------------------------------------------
class PointIn(BaseModel):
    x: float = Field(description="x coordinate of the point")
    y: float = Field(description="y coordinate of the point")


class BatchIn(BaseModel):
    points: list[PointIn] = Field(
        min_length=1,
        max_length=1000,
        description="1 to 1000 points to classify in one request",
    )


# ---- Responses -----------------------------------------------------------
class PredictionOut(BaseModel):
    class_id: int
    class_name: str
    probabilities: list[float]


class BatchOut(BaseModel):
    count: int
    predictions: list[PredictionOut]


class HealthOut(BaseModel):
    status: str
    model_version: str


class ImagePredictionOut(BaseModel):
    filename: str
    content_type: str
    width: int
    height: int
    mean_brightness: float
    predicted_label: str