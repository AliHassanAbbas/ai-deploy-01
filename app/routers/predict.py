"""All prediction endpoints. Protected by API key + rate limit."""
import asyncio
import io

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image, ImageStat, UnidentifiedImageError

from app.deps import get_predictor, rate_limit, verify_api_key
from app.schemas import BatchIn, BatchOut, ImagePredictionOut, PointIn, PredictionOut

router = APIRouter(
    prefix="/predict",
    tags=["predictions"],
    # These run before EVERY endpoint in this router:
    dependencies=[Depends(verify_api_key), Depends(rate_limit)],
)


# ---- Single prediction --------------------------------------------------
# Plain `def` on purpose: model inference is CPU work. FastAPI runs `def`
# endpoints in a worker thread, so the event loop is never blocked.
@router.post("", response_model=PredictionOut)
def predict(body: PointIn, predictor=Depends(get_predictor)):
    return predictor.predict([body.x, body.y])[0]


# ---- Batch prediction ------------------------------------------------------
@router.post("/batch", response_model=BatchOut)
def predict_batch(body: BatchIn, predictor=Depends(get_predictor)):
    points = [[p.x, p.y] for p in body.points]
    results = predictor.predict(points)   # ONE tensor pass for the whole list
    return {"count": len(results), "predictions": results}


# ---- Streaming prediction (ChatGPT-style, word by word) ---------------------
@router.get("/stream")
async def predict_stream(x: float, y: float, predictor=Depends(get_predictor)):
    result = predictor.predict([x, y])[0]
    confidence = result["probabilities"][result["class_id"]]
    text = (
        f"Analyzing point ({x}, {y}) ... "
        f"the model assigns {confidence:.2%} probability "
        f"to class '{result['class_name']}'. "
        f"Final answer: {result['class_name']}."
    )

    async def token_stream():
        for word in text.split(" "):
            yield word + " "
            await asyncio.sleep(0.15)   # simulate token-by-token generation

    return StreamingResponse(token_stream(), media_type="text/plain")


# ---- Image upload (the pattern every CV model API uses) ----------------------
@router.post("/image", response_model=ImagePredictionOut)
def predict_image(file: UploadFile = File(...)):
    if file.content_type not in ("image/png", "image/jpeg"):
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{file.content_type}'. Send PNG or JPEG.",
        )

    data = file.file.read()
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="File is not a valid image.")

    # Toy "CV model": mean brightness -> bright/dark. A real model would
    # preprocess `image` to a tensor and call predictor.predict() here.
    grayscale = image.convert("L")
    brightness = ImageStat.Stat(grayscale).mean[0]
    label = "bright_image" if brightness >= 128 else "dark_image"

    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "width": image.width,
        "height": image.height,
        "mean_brightness": round(brightness, 2),
        "predicted_label": label,
    }