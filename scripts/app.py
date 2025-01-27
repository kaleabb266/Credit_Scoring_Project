from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd

# Load the trained model
model_path = "../models/Tuned_Random_Forest.pkl"
try:
    model = joblib.load(model_path)
except FileNotFoundError:
    raise FileNotFoundError(f"Model file not found at {model_path}. Ensure the model is trained and saved.")

# Initialize the FastAPI app
app = FastAPI()

# Define the request body structure
class PredictionInput(BaseModel):
    TransactionHour: int
    TransactionDay: int
    TransactionMonth: int
    TransactionYear: int
    TransactionsPerDay: float
    TimeSinceLastTransaction: float
    Amount: float
    Value: int
    TotalTransactionAmount: float
    AverageTransactionAmount: float
    TransactionCount: int
    TransactionStdDev: float
    Recency: int
    Frequency: int
    Monetary: float
    Seasonality: int
    RFMS_Score: float
    RFMS_Cluster: int

# API root endpoint
@app.get("/")
def root():
    return {"message": "Welcome to the Credit Scoring Model API!"}

# Prediction endpoint
@app.post("/predict")
def predict(input_data: PredictionInput):
    try:
        # Convert input data to a DataFrame
        input_df = pd.DataFrame([input_data.dict()])
        
        # Make predictions
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0].tolist()
        
        # Map prediction to a label
        label = "Fraudulent" if prediction == 1 else "Good"
        
        # Return the prediction and probabilities
        return {
            "prediction": label,
            "probability": {
                "Good": probability[0],
                "Fraudulent": probability[1]
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
