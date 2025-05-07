# data_loader.py

import pandas as pd
from sklearn.model_selection import train_test_split

def load_csv(path, target_column=None, test_size=0.2, random_state=42):
    df = pd.read_csv(r"C:\Users\retyr\Desktop\GitHub Projects\IN-WORK\ai_automl_project\data\Airline_Ticket_Price_data.csv")

    if target_column and target_column in df.columns:
        X = df.drop(columns=[target_column])
        y = df[target_column]
        return train_test_split(X, y, test_size=test_size, random_state=random_state)
    else:
        return df
