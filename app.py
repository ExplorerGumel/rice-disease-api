from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import uvicorn
import os
from inference import load_model, predict

MODEL_PATH = os.getenv("MODEL_PATH", "models/efficientnet_b0_best.pth")
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
    description="Detects rice leaf diseases from images using EfficientNet-B0",
    version="1.0.0",
    lifespan=lifespan,
)

@app.get("/")
def home():
    return {
        "message": "Rice Leaf Disease Detection API",
        "version": "1.0.0",
        "docs":    "/docs",
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model":  "EfficientNet-B0",
        "classes": ["bacterial", "blast", "brownspot", "healthy", "tungro"],
    }

@app.post("/predict")
async def predict_disease(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="File must be an image (jpg, png, etc.)"
        )
    try:
        image_bytes = await file.read()
        result      = predict(model, image_bytes)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)