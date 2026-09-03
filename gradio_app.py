import io

import gradio as gr

from inference import load_model, predict


MODEL_PATH = "models/efficientnet_b0_best.pth"

print(f"Loading model from {MODEL_PATH}...")

model = load_model(MODEL_PATH)

print("✅ Model loaded successfully")


def predict_disease(image):
    if image is None:
        return {"error": "Please upload a rice leaf image."}

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")

    result = predict(model, buffer.getvalue())

    return result


demo = gr.Interface(
    fn=predict_disease,
    inputs=gr.Image(
        type="pil",
        label="Upload Rice Leaf Image"
    ),
    outputs=gr.JSON(
        label="Prediction"
    ),
    title="🌾 Rice Leaf Disease Detector",
    description=(
        "Upload a rice leaf image to classify its disease "
        "using an EfficientNet-B0 deep learning model."
    )
)


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )
