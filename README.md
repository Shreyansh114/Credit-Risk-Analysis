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
