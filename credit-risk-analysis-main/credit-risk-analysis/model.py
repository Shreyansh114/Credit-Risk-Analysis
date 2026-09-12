import joblib
import os
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, classification_report

def train(X_train, y_train, model_path='models/xgb_model.joblib'):
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    model.fit(X_train, y_train)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    return model

def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:,1]
    return {
        'accuracy': accuracy_score(y_test, preds),
        'roc_auc': roc_auc_score(y_test, probs),
        'confusion_matrix': confusion_matrix(y_test, preds).tolist(),
        'classification_report': classification_report(y_test, preds, output_dict=True)
    }

if __name__ == '__main__':
    # quick run script
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    df = pd.read_csv('data/sample_loan_data.csv')
    X = df.drop(columns=['default'])
    y = df['default']
    scaler = StandardScaler()
    X[X.select_dtypes(include=['int64','float64']).columns] = scaler.fit_transform(X.select_dtypes(include=['int64','float64']))
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    model = train(X_train, y_train)
    print('Evaluation:', evaluate(model, X_test, y_test))
