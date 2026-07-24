"""Application entry point. Run from the project root:

    uvicorn app.main:app --reload
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.routers import predict, system
from src.config import BUNDLE_DIR
from src.predictor import Predictor


# ---- Lifespan: load the model ONCE at startup ----------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.predictor = Predictor(BUNDLE_DIR)
    print(f"model loaded: version {app.state.predictor.config['model_version']}")
    yield
    app.state.predictor = None
    print("model released")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    lifespan=lifespan,
)

# ---- CORS: which websites' JavaScript may call this API --------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],   # your future frontend
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Routers -----------------------------------------------------------------
app.include_router(system.router)
app.include_router(predict.router)


# ---- Global safety net: NO request may crash the server ------------------------
@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    # Log the full detail for YOU...
    print(f"UNHANDLED ERROR on {request.method} {request.url.path}: {exc!r}")
    # ...but send the client a clean, generic JSON (never a stack trace).
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. The incident has been logged."},
    )