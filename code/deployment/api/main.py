# code/deployment/api/main.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import uvicorn
from pathlib import Path

current_directory = Path.cwd()
path = current_directory / Path("models/catboost.pkl")
# Load the trained model (modify the path and filename as needed)
model = joblib.load(path)

app = FastAPI()  # Create an instance of the FastAPI class


# Define a request model that will be used to validate incoming data
class PredictionRequest(BaseModel):
    features: list


# Define an endpoint for model predictions
@app.post("/predict/")
async def predict(request: PredictionRequest):
    # Convert the input features into a numpy array
    print(1)
    input_features = np.array(request.features).reshape(1, -1)
    print(2)
    # Make a prediction
    prediction = model.predict(input_features)
    print(prediction[0])
    # Return the prediction as JSON
    return {"prediction": str(prediction[0])}

# Run the API if this script is executed directly (useful for testing)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
