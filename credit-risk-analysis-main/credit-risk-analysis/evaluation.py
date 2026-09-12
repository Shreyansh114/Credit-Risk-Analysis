import matplotlib.pyplot as plt
import pandas as pd
import joblib
from sklearn.metrics import roc_curve, auc
import os

def plot_roc(model, X_test, y_test, out_path='reports/roc_curve.png'):
    probs = model.predict_proba(X_test)[:,1]
    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0,1], [0,1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc='lower right')
    plt.savefig(out_path)
    plt.close()
    return out_path

if __name__ == '__main__':
    df = pd.read_csv('data/sample_loan_data.csv')
    X = df.drop(columns=['default'])
    y = df['default']
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X[X.select_dtypes(include=['int64','float64']).columns] = scaler.fit_transform(X.select_dtypes(include=['int64','float64']))
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    import joblib
    from model import train, evaluate
    model = train(X_train, y_train)
    print('Saved ROC at', plot_roc(model, X_test, y_test))
