from fastapi import FastAPI, HTTPException, File, UploadFile
from tensorflow.keras.applications.densenet import DenseNet121, preprocess_input, decode_predictions
from PIL import Image
import numpy as np
import io
from app.model import load_model, load_model_from_package, predict, preprocess_image

app = FastAPI()

# Load, in memory, the pre-trained DenseNet121 model with ImageNet weights
###########################################################################
# 1- From tensorflow library :
model = load_model_from_package()
# 2- OR From google link online :
model = load_model()
model.trainable = False  # Freeze the model for inference


# POST/predict endpoint. 
############################################################################
#       Request  
#       - Form-data with key 'file' containing an image file
#       Response 
#       - JSON object with the DenseNet121 predicted class label 
@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    """
    Handles image upload, processes it, and returns the model's prediction.
    """
    try:
        # Read the uploaded image
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")  # Ensure it's in RGB format

        # Preprocess the image
        processed_img = preprocess_image(img)

        # Run prediction
        predictions = model.predict(processed_img)

        # Decode predictions into human-readable labels
        decoded_preds = decode_predictions(predictions, top=1)[0][0][1]  # Get top-1 class

        return {"prediction": decoded_preds}
    
    except IOError:
        raise HTTPException(status_code=400, detail="Invalid image format or corrupted file.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")



# Optionnal, just for me to check if everything is OK
@app.get("/status")
def get_status():
    return {"status": "Server is running"}