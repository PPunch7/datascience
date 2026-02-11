import streamlit as st
import joblib
import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.load_data import load_forex_data
from src.preprocess import clean_missing_values, make_ml_features
from src.save_model import BaselineNaiveModel

# ======================
# CONFIG
# ======================
DATA_PATH = "data/Foreign_Exchange_Rates.csv"
MODELS_DIR = Path("models")

# ======================
# UI
# ======================
st.title("FX Forecast App")
model_files = list(MODELS_DIR.glob("*_meta.json"))
if not model_files:
    st.error("No trained models found")
    st.stop()
currencies = [f.stem.replace("_meta","").replace("_","/") for f in model_files]
currency = st.selectbox("Select currency", currencies)
horizon = st.slider("Forecast days", 5, 60, 30)
run_btn = st.button("Run Forecast")

# ======================
# LOAD DATA
# ======================
@st.cache_data
def load_data():
    df = load_forex_data(DATA_PATH)
    df = clean_missing_values(df)
    return df

df = load_data()

# ======================
# LOAD MODEL META
# ======================
meta_path = MODELS_DIR / f"{currency.replace('/','_')}_meta.json"
with open(meta_path) as f:
    meta = json.load(f)
best_model_name = meta["best_model"]
feature_cols = meta["feature_cols"]
st.write("Best model:", best_model_name)

# ======================
# LOAD MODEL OBJECT
# ======================
def load_model(currency, model_name):
    base = currency.replace("/","_")

    # keras
    keras_path = MODELS_DIR / f"{base}_{model_name}.keras"
    if keras_path.exists():
        from tensorflow.keras.models import load_model
        return load_model(keras_path), "keras"

    # pickle / joblib
    pkl_path = MODELS_DIR / f"{base}_{model_name}.pkl"
    if pkl_path.exists():
        return joblib.load(pkl_path), "sklearn"

    raise ValueError("Model file not found")

model, model_type = load_model(currency, best_model_name)

# ======================
# FORECAST
# ======================
if run_btn:
    st.subheader("Forecast")
    series_df = df[["Date", currency]].copy()

    # build ML features
    feat_df, feat_cols = make_ml_features(series_df, currency)
    feat_df = feat_df.dropna()

    X = feat_df.reindex(columns=feature_cols, fill_value=0)
    X_last = X.tail(horizon)

    # -------- predict --------
    if isinstance(model, BaselineNaiveModel):
        preds = model.predict(X_last)
    elif model_type == "sklearn":
        preds = model.predict(X_last)
    elif model_type == "keras":
        WINDOW = 20
        vals = series_df[currency].values
        seq = vals[-(WINDOW+horizon):]

        X_lstm = []
        for i in range(WINDOW, len(seq)):
            X_lstm.append(seq[i-WINDOW:i])

        X_lstm = np.array(X_lstm).reshape((-1, WINDOW, 1))
        preds = model.predict(X_lstm).flatten()

    else:
        st.error("Unknown model type")
        st.stop()

    # ======================
    # PLOT
    # ======================
    actual = series_df[currency].tail(horizon).values
    fig, ax = plt.subplots(figsize=(10,4))

    ax.plot(actual, label="Actual")
    ax.plot(preds, label="Forecast")

    ax.set_title(currency)
    ax.legend()

    st.pyplot(fig)
    out = pd.DataFrame({
        "actual": actual,
        "forecast": preds
    })

    st.dataframe(out)
