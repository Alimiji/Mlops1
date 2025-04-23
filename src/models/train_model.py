# Entraînement, sauvegarde, évaluation


import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

def train():
    df = pd.read_csv("data/processed/train.csv")
    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    joblib.dump(model, "models/model.pkl")
    print("✅ Modèle entraîné et sauvegardé")

if __name__ == "__main__":
    train()

