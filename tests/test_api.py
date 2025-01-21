from fastapi.testclient import TestClient
from app.main import app
import base64
import os

client = TestClient(app)

def test_status_endpoint():
    """Test the status endpoint to check if the server is running."""
    response = client.get("/status")
    assert response.status_code == 200
    assert response.json() == {"status": "Server is running"}

def test_predict_endpoint():
    """Test the image prediction endpoint with a sample image."""
    test_image_path = os.path.join(os.path.dirname(__file__), "Req_image.jpg")
    
    # Read and encode image
    with open(test_image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")

    # Send a request with the encoded image
    response = client.post("/predict", files={"file": open(test_image_path, "rb")})

    print(response.status_code)
    print(response.json())

    # Assert the response status and check the prediction
    assert response.status_code == 200
    assert "prediction" in response.json()
