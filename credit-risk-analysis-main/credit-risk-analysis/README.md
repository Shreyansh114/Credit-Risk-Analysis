# Credit Risk Analysis Model

**Overview**
This repository contains a credit risk model pipeline: data preprocessing, training (XGBoost), evaluation and a simple FastAPI prediction endpoint.
A small synthetic sample dataset is included (`data/sample_loan_data.csv`) so you can run and test locally.

**Structure**
- `data/` - sample dataset
- `models/` - saves trained model and scaler (created after training)
- `preprocessing.py` - data preprocessing utilities
- `model.py` - training & evaluation scripts
- `evaluation.py` - plotting utilities
- `app.py` - FastAPI app to serve predictions
- `requirements.txt` - Python dependencies
- `Dockerfile` - containerize the app

**Quick start**
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
# Train model and save artifacts
python -c "from preprocessing import load_data, preprocess, save_scaler; df=load_data('data/sample_loan_data.csv'); (X_train, X_test, y_train, y_test), scaler=preprocess(df); from model import train; train(X_train, y_train); save_scaler(scaler, 'models/scaler.joblib')"
# Run FastAPI server
uvicorn app:app --reload --port 8000
```

**Notes**
- Replace `data/sample_loan_data.csv` with your own dataset (same column names) for real training.
- This scaffold is meant as a starting point — add hyperparameter tuning, SHAP explainability, Spark pipelines, and CI as next steps.
