from fastapi.testclient import TestClient
from app import app
import io
from PIL import Image

with TestClient(app) as client:
    def test_home():
        response = client.get("/")
        assert response.status_code == 200
        assert "message" in response.json()

    def test_health():
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_predict_valid_image():
        img = Image.new("RGB", (224, 224), color=(100, 150, 100))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)
        response = client.post(
            "/predict",
            files={"file": ("test.jpg", buf, "image/jpeg")}
        )
        assert response.status_code == 200
        data = response.json()
        assert "predicted_class" in data

    def test_predict_invalid_file():
        response = client.post(
            "/predict",
            files={"file": ("test.txt", b"not an image", "text/plain")}
        )
        assert response.status_code == 400
