import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def encode_features(df, target_column):
    categorical_cols = df.select_dtypes(include='object').columns.tolist()
    categorical_cols = [col for col in categorical_cols if col != target_column]

    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    encoded = encoder.fit_transform(df[categorical_cols])

    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_cols), index=df.index)
    df = df.drop(columns=categorical_cols)
    df = pd.concat([df, encoded_df], axis=1)

    return df
