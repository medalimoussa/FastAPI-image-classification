# FastAPI DenseNet121 Image Classification


This project is a FastAPI-based web service for image classification using the pre-trained DenseNet121 deep learning model. The API allows users to upload images and receive predictions with confidence scores.


# Project Goals
Develop an efficient image classification API using FastAPI.
Utilize a pre-trained DenseNet121 model to classify images.
Provide endpoints for image predictions and health checks.
Ensure code maintainability with modular architecture.
Implement automated testing to guarantee reliability.

# Project structure
FastAPI project/
│-- app/
│   │-- __init__.py
│   │-- main.py               # API routes and app instance
│   │-- model.py              # Model loading and prediction logic
│   │-- schemas.py            # Pydantic request/response models
│   │-- utils.py              # Utility functions (e.g., image preprocessing)
│-- tests/
│   │-- __init__.py
│   │-- test_api.py            # Tests for API endpoints
│   │-- test_model.py          # Tests for model loading and predictions
│-- requirements.txt          # Python dependencies
│-- Dockerfile                # Docker configuration for deployment
│-- README.md                 # Project documentation


# Getting Started

1. Clone the Repository

git clone https://github.com/your-username/your-repo.git
cd your-repo


2. Set Up a Virtual Environment
It is recommended to create a virtual environment to isolate project dependencies.
python -m venv venv
source venv/bin/activate   # On Linux/macOS
venv\Scripts\activate      # On Windows


3. Install Dependencies
Install all required Python packages from the requirements.txt file.
pip install -r requirements.txt


# Testing the API

1- Via test client 
   1-1 To run all tests : 
      $env:PYTHONPATH = $PWD
      pytest tests/
   1-2 To test specific files
      pytest tests/test_api.py
      pytest tests/test_model.py

2- Via Uvicorn for 

   2-1 run 
   uvicorn app.main:app --host 127.0.0.1 --port 5000 --reload
   2-2 Open the terminal in the root and run :
   curl -X POST "http://localhost:5000/predict" -F "file=@tests/Req_image.jpg"


# Technologies Used
FastAPI – Modern web framework for building APIs with Python.
TensorFlow/Keras – Deep learning framework for model inference.
Pillow (PIL) – Image processing library.
Uvicorn – ASGI server for serving FastAPI applications.
Pydantic – Data validation and serialization library.
Pytest – Testing framework for unit and API tests.





## Installation



1. Clone the repository:
   ```bash
   git clone https://github.com/yourrepo/ML_Interview_Assignment.git
   cd ML_Interview_Assignment
