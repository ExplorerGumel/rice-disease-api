import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import io

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLASS_NAMES = ["bacterial", "blast", "brownspot", "healthy", "tungro"]

DISEASE_INFO = {
    "bacterial": {
        "full_name": "Bacterial Leaf Blight",
        "severity":  "High",
        "treatment": "Apply copper-based bactericides. Remove infected plants.",
    },
    "blast": {
        "full_name": "Rice Blast",
        "severity":  "Very High",
        "treatment": "Apply tricyclazole fungicide. Avoid excess nitrogen.",
    },
    "brownspot": {
        "full_name": "Brown Spot",
        "severity":  "Medium",
        "treatment": "Apply mancozeb fungicide. Improve soil nutrition.",
    },
    "healthy": {
        "full_name": "Healthy",
        "severity":  "None",
        "treatment": "No treatment needed.",
    },
    "tungro": {
        "full_name": "Rice Tungro Disease",
        "severity":  "Very High",
        "treatment": "Control leafhopper vectors. Remove infected plants.",
    },
}

TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


def load_model(model_path: str):
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model = models.efficientnet_b0(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(1280, len(CLASS_NAMES))
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    return model


def predict(model, image_bytes: bytes) -> dict:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = TRANSFORMS(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs     = model(tensor)
        probs       = torch.softmax(outputs, dim=1)[0]
        pred_idx    = probs.argmax().item()
        confidence  = round(probs[pred_idx].item(), 4)

    predicted_class = CLASS_NAMES[pred_idx]
    all_probs = {
        CLASS_NAMES[i]: round(probs[i].item(), 4)
        for i in range(len(CLASS_NAMES))
    }

    return {
        "predicted_class": predicted_class,
        "confidence":      confidence,
        "all_probabilities": all_probs,
        "disease_info":    DISEASE_INFO[predicted_class],
    }