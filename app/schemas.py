from pydantic import BaseModel
from typing import Optional


#For the input, since we are giving the filename we do not require to have a request schemas

# Response schema for prediction results (if we decided to use it)
class PredictionResponse(BaseModel):
    prediction: str  # Predicted class label 
    confidence: Optional[float] = None  # Confidence score (if available)
    
# In this project we are not gonna use this file because we are getting only the prediction
# label which is a simple information. In a more bigger project it could be considred.
