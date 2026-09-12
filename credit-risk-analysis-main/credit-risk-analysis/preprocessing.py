import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

def load_data(path):
    return pd.read_csv(path)

def preprocess(df, target='default'):
    X = df.drop(columns=[target])
    y = df[target]
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
    return train_test_split(X, y, test_size=0.2, random_state=42), scaler

def save_scaler(scaler, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    joblib.dump(scaler, out_path)

if __name__ == '__main__':
    df = load_data('data/sample_loan_data.csv')
    (X_train, X_test, y_train, y_test), scaler = preprocess(df)
    save_scaler(scaler, 'models/scaler.joblib')
    print('Preprocessing done. Train shape:', X_train.shape)
