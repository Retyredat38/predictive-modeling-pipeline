# preprocess.py

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

def build_preprocessing_pipeline(X: pd.DataFrame):
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numerical_cols = X.select_dtypes(include=["number"]).columns.tolist()

    # Pipelines
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    # Full pipeline
    full_pipeline = ColumnTransformer([
        ("cat", categorical_pipeline, categorical_cols),
        ("num", numeric_pipeline, numerical_cols)
    ])

    return full_pipeline, categorical_cols, numerical_cols
