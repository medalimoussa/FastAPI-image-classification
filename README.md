
# FastAPI DenseNet121 Image Classification

This project is a FastAPI-based web service for image classification using the pre-trained DenseNet121 deep learning model. The model can be loaded by one of two ways. The API allows users to upload images and receive predictions. 
  
## Project structure

FastAPI project/

│-- **app**/

│ │--  \_\_init\_\_.py

│ │-- main.py # API routes and app instance

│ │-- model.py # Model loading and prediction logic

│ │-- schemas.py # Pydantic request/response models

│ │-- utils.py # Utility functions (e.g., image preprocessing)

│-- tests/

│ │--  \_\_init\_\_.py

│ │-- test_api.py # Tests for API endpoints

│ │-- test_model.py # Tests for model loading and predictions

│ │-- Req_image.jpg # The used image for the testing 

│-- requirements.txt # Python dependencies

│-- README.md # Project documentation



## 1- Getting Started

  
### 1-1- Clone the Repository

`git clone https://github.com/your-username/your-repo.git`
`cd your-repo`

### 1.2- Set Up a Virtual Environment

It is recommended to create a virtual environment to isolate project dependencies.

 `python -m venv venv `

 `source venv/bin/activate` # if you are in Linux/macOS

 `venv\Scripts\activate` # if you are in Windows

### 1.3- Install the needed Dependencies

Install all required Python packages from the __requirements.txt__ file.

 `pip install -r requirements.txt `

  
  

## 2- Testing the API

To test the API, we can either launch a FastAPI server using **Uvicorn**, and then send requests to the running instance, or use the **TestClient**, which allows us to run tests without starting an actual server.

We can also test using a RUST client interface. For this case, the image must be transformed into a base64 and input it into the body of the request we want to send.

### 2.1- By using TestClient

&rarr; To run all tests, in a cmd run  :

&nbsp;&nbsp;&nbsp;&nbsp; `$env:PYTHONPATH = $PWD`

&nbsp;&nbsp;&nbsp;&nbsp; `pytest tests/`

&rarr; To test specific files (either the api or the model)

&nbsp;&nbsp;&nbsp;&nbsp; `pytest tests/test_api.py`

&nbsp;&nbsp;&nbsp;&nbsp; `pytest tests/test_model.py`

  

### 2.2- Via Uvicorn

Run the following command in order to launch the FastAPI application
`uvicorn app.main:app --host 127.0.0.1 --port 5000 --reload`

Open a new the terminal in **the root directory** (`FastAPI project/`) and run :

`curl -X POST "http://localhost:5000/predict" -F "file=@tests/Req_image.jpg"`

&rarr;  Upload an image of your choice and change `tests/Req_image.jpg` with an image path and name.
  Note : the result depend on the model weights.
  

## Technologies Used

Which are put in the **requirement.txt** file for installation in the dependency part. :

* FastAPI – The used modern web framework for building APIs with Python.

* TensorFlow/Keras –  Deep learning framework for model inference.

* Pillow (PIL) – Image processing library.

* Uvicorn – ASGI server for serving FastAPI applications.

* Pydantic – Data validation and serialization library.

* Pytest – Testing framework for unit and API tests.

## Author [Mohamed Alimoussa](https://gist.github.com/Mohamed111995)
