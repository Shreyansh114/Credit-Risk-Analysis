from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title='Credit Risk Predictor')

class LoanRequest(BaseModel):
    age: int
    income: float
    loan_amount: float
    loan_term: int
    credit_history: int
    num_dependents: int

model = None
scaler = None

@app.on_event('startup')
def load():
    global model, scaler
    model = joblib.load('models/xgb_model.joblib')
    scaler = joblib.load('models/scaler.joblib')

@app.post('/predict')
def predict(req: LoanRequest):
    data = pd.DataFrame([req.dict()])
    numeric_cols = data.select_dtypes(include=['int64','float64']).columns.tolist()
    data[numeric_cols] = scaler.transform(data[numeric_cols])
    prob = model.predict_proba(data)[:,1][0]
    pred = int(prob > 0.5)
    return {'probability_default': float(prob), 'predicted_default': pred}
