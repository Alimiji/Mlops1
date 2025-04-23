# Scripts de suivi du modèle (drift, alertes)

import pandas as pd
import joblib
from sklearn.metrics import accuracy_score

def monitor():
    model = joblib.load("models/model.pkl")
    df = pd.read_csv("data/processed/monitor.csv")
    X = df.drop("target", axis=1)
    y_true = df["target"]
    y_pred = model.predict(X)

    acc = accuracy_score(y_true, y_pred)
    print(f"🧪 Current model accuracy on live data: {acc:.4f}")

if __name__ == "__main__":
    monitor()

