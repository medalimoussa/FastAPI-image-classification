import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications.densenet import DenseNet121, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os

# URL of the pre-trained model weights
MODEL_URL = "https://storage.googleapis.com/tensorflow/keras-applications/densenet/densenet121_weights_tf_dim_ordering_tf_kernels.h5"
MODEL_FILENAME = "densenet121_weights.h5"

# Function to download the model if not available locally
def download_model():
    model_path = os.path.join(os.path.dirname(__file__), MODEL_FILENAME)

    if not os.path.exists(model_path):
        print("Downloading DenseNet121 model weights...")
        model_path = tf.keras.utils.get_file(
            fname=MODEL_FILENAME,
            origin=MODEL_URL,
            cache_dir=os.path.dirname(__file__),  # Download into current directory
            cache_subdir="."
        )   
        print(f"Model downloaded to: {model_path}")
    else:
        print("Model weights already exist. Skipping download.")
    
    return model_path

# Function to load the model
def load_model():
    model_path = download_model()
    model = DenseNet121(weights=model_path)
    model.trainable = False  # Freeze the model
    return model

# Function to load the model without the need to download the weighths
def load_model_from_package():
    model = DenseNet121(weights="imagenet")
    model.trainable = False  # Freeze the model for inference


# Image preprocessing function
def preprocess_image(image: Image.Image):
    """
    Preprocess an image for DenseNet121 model input.
    - Resizes the image to (224, 224).
    - Converts the image to a NumPy array.
    - Applies the necessary preprocessing transformations.
    """
    img = image.resize((224, 224))  # Resize to 224x224 (required for DenseNet121)
    img_array = np.array(img)  # Convert to NumPy array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = preprocess_input(img_array)  # Normalize image
    return img_array

# Function to preprocess the image and make predictions
def predict(model, img):
    img_array = preprocess_image(img)

    predictions = model.predict(img_array)
    decoded_predictions = decode_predictions(predictions, top=1)[0][0][1]  # Get top predicted class
    return decoded_predictions
