# Scripts de traitement et chargement
# Structure d'un projet MLOps type

# Fichier: src/data/make_dataset.py
import pandas as pd
from sklearn.model_selection import train_test_split
import os

def prepare_data():
    df = pd.read_csv("data/raw/bank-additional-full.csv", sep=';')

    # Simplification : transformer la cible en binaire
    df['y'] = df['y'].map({'yes': 1, 'no': 0})

    # Exemple de traitement minimal : conversion des variables catégorielles
    df = pd.get_dummies(df, drop_first=True)

    train_df, monitor_df = train_test_split(df, test_size=0.2, random_state=42)

    os.makedirs("data/processed", exist_ok=True)
    train_df.to_csv("data/processed/train.csv", index=False)
    monitor_df.to_csv("data/processed/monitor.csv", index=False)

    print("✅ Données préparées et sauvegardées.")

if __name__ == "__main__":
    prepare_data()

# Fichier: src/models/train_model.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

def train():
    df = pd.read_csv("data/processed/train.csv")
    X = df.drop("y", axis=1)
    y = df["y"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    joblib.dump(model, "models/model.pkl")
    print("✅ Modèle entraîné et sauvegardé")

if __name__ == "__main__":
    train()


# Fichier: src/monitoring/monitor_model.py
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score

def monitor():
    model = joblib.load("models/model.pkl")
    df = pd.read_csv("data/processed/monitor.csv")
    X = df.drop("y", axis=1)
    y_true = df["y"]
    y_pred = model.predict(X)

    acc = accuracy_score(y_true, y_pred)
    print(f"🧪 Précision actuelle du modèle sur les données en ligne : {acc:.4f}")

if __name__ == "__main__":
    monitor()


# Fichier: config/params.yaml
"""
train:
  test_size: 0.2
  random_state: 42
  n_estimators: 100

paths:
  raw_data: data/raw/bank-additional-full.csv
  train_data: data/processed/train.csv
  monitor_data: data/processed/monitor.csv
  model: models/model.pkl
"""

# Fichier: requirements.txt
"""
pandas
scikit-learn
joblib
yaml
"""

# Fichier: README.md
"""

"""

