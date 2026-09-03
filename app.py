from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import uvicorn
import os

from inference import load_model, predict


MODEL_PATH = os.getenv(
    "MODEL_PATH",
    "models/efficientnet_b0_best.pth"
)

model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model

    print(f"Loading model from {MODEL_PATH}...")

    model = load_model(MODEL_PATH)

    print("✅ Model loaded successfully")

    yield

    print("Shutting down...")


app = FastAPI(
    title="Rice Leaf Disease Detection API",
    description="AI-powered rice leaf disease classification using EfficientNet-B0",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)


@app.get("/")
def home():
    return {
        "message": "Rice Leaf Disease Detection API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


@app.post("/predict")
async def predict_disease(file: UploadFile = File(...)):

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="File must be an image"
        )

    image_bytes = await file.read()

    result = predict(model, image_bytes)

    return JSONResponse(content=result)


if __name__ == "__main__":
    port = int(os.getenv("PORT", 7860))

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        reload=False
    )
