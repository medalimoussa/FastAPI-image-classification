from app.model import load_model, predict
from PIL import Image
import os

def test_model_loading():
    """Test that the model loads correctly without errors."""
    model = load_model()
    assert model is not None, "Model failed to load"

def test_model_prediction():
    """Test the model prediction with a sample image."""
    model = load_model()
    test_image_path = os.path.join(os.path.dirname(__file__), "Req_image.jpg")

    # Load test image
    img = Image.open(test_image_path).convert("RGB")

    # Perform prediction
    prediction_result = predict(model, img)

    # Check the prediction result
    assert isinstance(prediction_result, str), "Prediction result should be a string"
    assert len(prediction_result) > 0, "Prediction should not be empty"
