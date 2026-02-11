import json
from pathlib import Path
import joblib
import pickle
from tensorflow.keras.models import save_model as keras_save_model

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

def save_sklearn_model(model, currency: str, model_name: str):
    fname = f"{currency.replace('/','_')}_{model_name}.pkl"
    path = MODELS_DIR / fname
    joblib.dump(model, path)
    print(f"[Saved] {path}")
    return path

def save_pickle_model(model, currency: str, model_name: str):
    path = MODELS_DIR / f"{currency.replace('/','_')}_{model_name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"[Saved] {path}")
    return path

def save_keras_model(model, currency: str, model_name: str):
    path = MODELS_DIR / f"{currency.replace('/','_')}_{model_name}.keras"
    keras_save_model(model, path)
    print(f"[Saved] {path}")
    return path

def save_metadata(currency, best_model_name, metrics_dict, feature_cols):
    meta = {
        "currency": currency,
        "best_model": best_model_name,
        "metrics": metrics_dict,
        "feature_cols": feature_cols
    }
    path = MODELS_DIR / f"{currency.replace('/','_')}_meta.json"
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[Saved] {path}")

def load_model(currency, model_name):
    path = MODELS_DIR / f"{currency.replace('/','_')}_{model_name}.pkl"
    return joblib.load(path)

class BaselineNaiveModel:

    def __init__(self, target_col):
        self.target_col = target_col

    def predict(self, X):
        lag_cols = [c for c in X.columns if "_lag_1" in c]
        if not lag_cols:
            raise ValueError("No lag_1 feature found")
        return X[lag_cols[0]].values