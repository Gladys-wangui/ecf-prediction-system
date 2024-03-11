# main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib

app = FastAPI()

# Load the pre-trained machine learning model
model = joblib.load('trained_model.joblib')

class AnimalSymptoms(BaseModel):
    temperature: float
    swollen_lymph_nodes: int
    loss_of_appetite: int
    weakness: int
    coughing: int
    rapid_breathing: int
    nasal_discharge: int
    anaemia: int

@app.post("/predict")
async def predict(symptoms: AnimalSymptoms):
    try:
        # Extract features from request
        features = [
            symptoms.temperature,
            symptoms.swollen_lymph_nodes,
            symptoms.loss_of_appetite,
            symptoms.weakness,
            symptoms.coughing,
            symptoms.rapid_breathing,
            symptoms.nasal_discharge,
            symptoms.anaemia,
        ]

        # Make predictions using the trained model
        prediction = model.predict([features])

        # Assuming your model returns binary predictions (0 or 1)
        result = "Positive" if prediction[0] == 1 else "Negative"

        return {"result": result}

    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid input. Please enter valid values for input features.")

