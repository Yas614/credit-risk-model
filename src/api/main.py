from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# Your data structures
class PredictionRequest(BaseModel):
    total_amount: float
    avg_amount: float
    transaction_count: int

class PredictionResponse(BaseModel):
    risk_probability: float

# Your endpoint
@app.post("/predict", response_model=PredictionResponse)
def predict_risk(request: PredictionRequest):

    return {"risk_probability": 0.27}