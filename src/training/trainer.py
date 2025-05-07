from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

def train_and_save_models(X, y, outdir="models"):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "RandomForest": RandomForestClassifier(n_estimators=100),
        "LogisticRegression": LogisticRegression(max_iter=1000)
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[name] = classification_report(y_test, y_pred, output_dict=True)
        joblib.dump(model, f"{outdir}/{name}.pkl")

    return results
