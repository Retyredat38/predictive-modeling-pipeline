# src/model_trainer.py

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb

def train_models(X_train, y_train, X_val, y_val):
    models = {
        "RandomForest": RandomForestRegressor(),
        "Ridge": Ridge(),
        "XGBoost": xgb.XGBRegressor()
    }

    param_distributions = {
        "RandomForest": {
            "n_estimators": [100, 200, 300],
            "max_depth": [None, 10, 20]
        },
        "Ridge": {
            "alpha": [0.01, 0.1, 1.0, 10.0]
        },
        "XGBoost": {
            "n_estimators": [100, 200],
            "max_depth": [3, 6, 10],
            "learning_rate": [0.01, 0.1, 0.3]
        }
    }

    best_model = None
    best_score = float("inf")
    best_model_name = None

    for name, model in models.items():
        search = RandomizedSearchCV(model, param_distributions[name], n_iter=5, cv=3, scoring="neg_mean_squared_error", random_state=42)
        search.fit(X_train, y_train)
        preds = search.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, preds))

        print(f"{name} RMSE: {rmse:.3f}, R²: {r2_score(y_val, preds):.3f}")

        if rmse < best_score:
            best_model = search.best_estimator_
            best_score = rmse
            best_model_name = name

    print(f"\nBest model: {best_model_name} with RMSE: {best_score:.3f}")
    return best_model
