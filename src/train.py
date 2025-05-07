
import os
import sys
import time
import pickle
import optuna
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from joblib import dump
from src.preprocessing.cleaner import preprocess_data

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

DATA_PATH = r"C:\Users\retyr\Desktop\GitHub Projects\IN-WORK\ai_automl_project\data\Airline_Ticket_Price_data.csv"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def train_models(X, y):
    models = {
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "Ridge": Ridge(alpha=1.0),  # Ridge doesn't support n_jobs
        "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42, n_jobs=-1)
    }

    best_score = float("inf")
    best_model = None
    best_name = ""

    for name, model in models.items():
        start = time.time()
        print(f"[INFO] Starting cross-validation for: {name}")
        try:
            score = -cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error').mean()
            print(f"[INFO] {name} CV RMSE: {score:.4f}")
            if score < best_score:
                best_score = score
                best_model = model
                best_name = name
        except Exception as e:
            print(f"[ERROR] Training failed for {name}: {e}")
        duration = time.time() - start
        print(f"[INFO] {name} training completed in {duration:.2f} seconds\n")

    if best_model:
        print(f"[INFO] Fitting best model: {best_name}")
        best_model.fit(X, y)
        model_path = os.path.join(MODEL_DIR, f"{best_name}_model.pkl")
        dump(best_model, model_path)
        print(f"[INFO] Best model saved: {best_name} with score {best_score:.4f}")
    else:
        print("[ERROR] No valid models were trained.")

def main():
    df = pd.read_csv(DATA_PATH)
    X, y = preprocess_data(df)
    train_models(X, y)

if __name__ == "__main__":
    main()
