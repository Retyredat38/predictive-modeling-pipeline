import pandas as pd

def preprocess_data(df):
    df_cleaned = df.copy()

    # Drop completely empty columns
    df_cleaned.dropna(axis=1, how='all', inplace=True)

    # Basic threshold for column drop (e.g. > 50% missing)
    threshold = 0.5
    df_cleaned.dropna(thresh=int(threshold * len(df_cleaned)), axis=1, inplace=True)

    # Fill numeric NaNs with median
    for col in df_cleaned.select_dtypes(include='number'):
        df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())


    # Fill non-numeric with mode
    for col in df_cleaned.select_dtypes(include='object'):
        df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mode()[0])

    # Extract target
    y = df_cleaned.pop('price')

    # One-hot encode object columns
    X = pd.get_dummies(df_cleaned)

    return X, y
