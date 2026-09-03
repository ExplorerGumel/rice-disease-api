import gradio as gr
from inference import load_model, predict


MODEL_PATH = "models/efficientnet_b0_best.pth"

print(f"Loading model from {MODEL_PATH}...")

model = load_model(MODEL_PATH)

print("✅ Model loaded successfully")


def predict_disease(image):
    if image is None:
        return {
            "error": "Please upload a rice leaf image."
        }

    # Convert Gradio image to PIL-compatible bytes
    import io

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
        "Upload an image of a rice leaf to classify "
        "the disease using EfficientNet-B0."
    ),
    submit_btn="Predict Disease",
    clear_btn="Clear"
)


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860
    )
