# Credit Risk Analysis

An end-to-end machine learning project for predicting the probability that a loan applicant will default. The project preprocesses loan data, trains an XGBoost classifier, evaluates performance, generates an ROC curve, and exposes predictions through a FastAPI service.

> The included dataset is synthetic and is intended for demonstration and testing only. It must not be used for real lending decisions.

## Features

- Preprocesses numeric loan-application features with `StandardScaler`
- Splits data into training and test sets
- Trains an XGBoost binary-classification model
- Evaluates accuracy, ROC-AUC, confusion matrix, and classification metrics
- Generates an ROC curve image
- Serves real-time predictions through a FastAPI endpoint
- Includes a Dockerfile for containerized deployment

## Project Structure

```text
credit-risk-analysis/
├── data/
│   └── sample_loan_data.csv    # Synthetic loan data
├── app.py                      # FastAPI prediction API
├── preprocessing.py            # Data loading and preprocessing
├── model.py                    # Model training and evaluation
├── evaluation.py               # ROC curve generation
├── requirements.txt            # Python dependencies
└── Dockerfile                  # Docker configuration
```

Dataset
The sample dataset contains the following input features:
Feature	Description
age	Applicant age
income	Applicant income
loan_amount	Requested loan amount
loan_term	Loan duration
credit_history	Credit-history indicator
num_dependents	Number of dependents
default	Target label: 1 for default, 0 for non-default


Installation
git clone <your-repository-url>
cd credit-risk-analysis

python -m venv venv
Activate the virtual environment:
# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
Install dependencies:
pip install -r requirements.txt
Train the Model
Run preprocessing to create and save the feature scaler:
python preprocessing.py
Then train the XGBoost model and print evaluation metrics:
python model.py
This creates:
models/scaler.joblib
models/xgb_model.joblib
Generate an ROC Curve
python evaluation.py
The ROC curve is saved to:
reports/roc_curve.png
Run the API
Start the FastAPI server after training the model:
uvicorn app:app --reload --port 8000
Open the interactive API documentation at:
http://127.0.0.1:8000/docs
Make a Prediction
Send a POST request to /predict:
{
  "age": 35,
  "income": 60000,
  "loan_amount": 15000,
  "loan_term": 36,
  "credit_history": 1,
  "num_dependents": 2
}
Example response:
{
  "probability_default": 0.12,
  "predicted_default": 0
}
predicted_default is 1 when the predicted default probability is greater than 0.5; otherwise it is 0.
Run with Docker
Build the image:
docker build -t credit-risk-analysis .
Run the container:
docker run -p 8000:8000 credit-risk-analysis
The API will be available at http://localhost:8000.
Technologies
- Python
- Pandas and NumPy
- Scikit-learn
- XGBoost
- FastAPI
- Uvicorn
- Matplotlib
- Docker
