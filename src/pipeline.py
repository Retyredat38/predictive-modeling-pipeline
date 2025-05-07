import os
from src.preprocessing.cleaner import load_and_clean_data
from src.preprocessing.encoder import encode_features
from src.training.trainer import train_and_save_models

def run_pipeline(csv_path, target_column):
    print("Loading and cleaning data...")
    df = load_and_clean_data(csv_path)

    print("Encoding categorical features...")
    df = encode_features(df, target_column)

    print("Splitting features and target...")
    X = df.drop(columns=[target_column])
    y = df[target_column]

    print("Training models...")
    os.makedirs("models", exist_ok=True)
    results = train_and_save_models(X, y)

    return results
