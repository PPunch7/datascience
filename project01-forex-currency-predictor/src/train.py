import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from autots import AutoTS

import matplotlib.pyplot as plt

from load_data import load_forex_data
from preprocess import (
    clean_missing_values,
    make_ml_features,
    make_lstm_dataset
)
from evaluate import (
    print_metrics,
    collect_metrics,
    print_direction_report,
    plot_confusion_multi,
    direction_summary_table,
    plot_model_predictions
)
from save_model import (
    save_sklearn_model, 
    save_pickle_model, 
    save_keras_model, 
    save_metadata, 
    load_model,
    BaselineNaiveModel
)

def load_and_prepare_data(path: str) -> pd.DataFrame:
    print("Loading raw data...")
    df = load_forex_data(path)

    print("Cleaning missing values (ND -> interpolate)...")
    df = clean_missing_values(df)

    # sanity checks
    if df["Date"].isna().any():
        raise ValueError("Date column contains NaT after parsing")

    if df.drop(columns="Date").isna().sum().sum() > 0:
        print("Still have missing — consider second pass interpolate")

    print("Data ready:", df.shape)
    return df

def time_series_split(df: pd.DataFrame, test_days: int):

    df = df.sort_values("Date").reset_index(drop=True)

    train = df.iloc[:-test_days].copy()
    test  = df.iloc[-test_days:].copy()

    print("\nSplit result:")
    print("Train:", train["Date"].min(), "->", train["Date"].max(), len(train))
    print("Test :", test["Date"].min(), "->", test["Date"].max(), len(test))

    return train, test

def build_ml_datasets(train_df, test_df, target, test_days):

    # ---------- TRAIN FEATURES ----------
    train_feat, feat_cols = make_ml_features(train_df, target)
    train_feat = train_feat.dropna(subset=feat_cols + [target])

    X_train = train_feat[feat_cols]
    y_train = train_feat[target]

    # ---------- TEST FEATURES ----------
    # buffer betwenn train and test data for calculating lag
    buffer = 20
    test_combo = pd.concat([train_df.tail(buffer), test_df])

    test_feat, _ = make_ml_features(test_combo, target)
    test_feat = test_feat.dropna(subset=feat_cols + [target])

    # split train/test
    test_start_date = test_df["Date"].min()
    test_feat = test_feat[test_feat["Date"] >= test_start_date]

    X_test = test_feat[feat_cols]
    y_test = test_feat[target]

    print("\nML dataset shapes:")
    print("X_train:", X_train.shape)
    print("X_test :", X_test.shape)

    print("\nDate check:")
    print("Train last date:", train_feat["Date"].max())
    print("Test first date:", test_feat["Date"].min())

    return X_train, X_test, y_train, y_test, feat_cols

def train_one_curr(df, target, test_days):
    print("\n" + "="*60)
    print("Training:", target)

    train_df, test_df = time_series_split(df, test_days)

    print("\nPreview train:")
    print(train_df.head())

    print("\nPreview test:")
    print(test_df.head())

    # ---- STEP 2.2 ----
    X_train, X_test, y_train, y_test, feat_cols = build_ml_datasets(
        train_df,
        test_df,
        target,
        test_days
    )

    # ---- STEP 2.3 ----
    results = []

    # BASELINE -> tomorrow = today
    print("\nBaseline...")
    y_pred_base = y_test.shift(1).bfill()
    baseline = BaselineNaiveModel(target)
    results.append(collect_metrics("BASELINE", y_test.values, y_pred_base.values))
    print_metrics("BASELINE", y_test.values, y_pred_base.values)

    ## XGBOOST ##
    print("\nTraining XGBoost...")
    xgb = XGBRegressor(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    xgb.fit(X_train, y_train)
    y_pred_xgb = xgb.predict(X_test)
    results.append(collect_metrics("XGBOOST", y_test.values, y_pred_xgb))
    print_metrics("XGBOOST", y_test.values, y_pred_xgb)

    # ---- STEP 2.4 ----
    ## LIGHTGBM ##
    print("\nTraining LightGBM...")
    lgb = LGBMRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=-1,
        random_state=42
    )
    lgb.fit(X_train, y_train)
    y_pred_lgb = lgb.predict(X_test)
    results.append(collect_metrics("LIGHTGBM", y_test.values, y_pred_lgb))
    print_metrics("LIGHTGBM", y_test.values, y_pred_lgb)

    ## ARIMA ##
    print("\nTraining ARIMA...")
    arima = ARIMA(train_df[target], order=(2,1,2)).fit()
    y_pred_arima = arima.forecast(test_days)
    results.append(collect_metrics("ARIMA", y_test.values, y_pred_arima.values))
    print_metrics("ARIMA", y_test.values, y_pred_arima.values)

    ## Prophet ##
    print("\nTraining Prophet...")
    p_train = train_df[["Date", target]].rename(
        columns={"Date": "ds", target: "y"}
    )
    prop = Prophet()
    prop.fit(p_train)
    future = prop.make_future_dataframe(periods=test_days)
    fcst = prop.predict(future)
    y_pred_prophet = fcst["yhat"].tail(test_days).values
    results.append(collect_metrics("PROPHET", y_test.values, y_pred_prophet))
    print_metrics("PROPHET", y_test.values, y_pred_prophet)

    ## LSTM ##
    # LSTM DATA
    WINDOW = 20
    train_series = train_df[target].values
    test_series  = test_df[target].values
    full_series = np.concatenate([train_series, test_series])
    scaler = StandardScaler()
    full_scaled = scaler.fit_transform(full_series.reshape(-1,1)).flatten()
    train_scaled = full_scaled[:-test_days]
    test_scaled  = full_scaled[-(test_days+WINDOW):]
    X_lstm_train, y_lstm_train = make_lstm_dataset(train_scaled, WINDOW)
    X_lstm_test,  y_lstm_test  = make_lstm_dataset(test_scaled, WINDOW)
    X_lstm_train = X_lstm_train.reshape((-1, WINDOW, 1))
    X_lstm_test  = X_lstm_test.reshape((-1, WINDOW, 1))

    print("\nTraining LSTM...")
    lstm = Sequential([
        LSTM(32, input_shape=(WINDOW,1)),
        Dense(1)
    ])
    lstm.compile(
        optimizer="adam",
        loss="mse"
    )
    lstm.fit(
        X_lstm_train,
        y_lstm_train,
        epochs=10,
        batch_size=32,
        verbose=0
    )

    y_pred_lstm_scaled = lstm.predict(X_lstm_test).flatten()
    y_pred_lstm = scaler.inverse_transform(
        y_pred_lstm_scaled.reshape(-1,1)
    ).flatten()
    y_true_lstm = scaler.inverse_transform(
        y_lstm_test.reshape(-1,1)
    ).flatten()
    results.append(collect_metrics("LSTM", y_true_lstm, y_pred_lstm))
    print_metrics("LSTM", y_true_lstm, y_pred_lstm)

    # ## AutoTS ##
    # print("\nTraining AutoTS...")
    # autots_df = df[["Date", target]].copy()
    # autots_df = autots_df.set_index("Date")
    # autots = AutoTS(
    #     forecast_length=test_days,
    #     frequency="infer",
    #     ensemble="simple",
    #     model_list="fast"
    # )
    # autots = autots.fit(
    #     autots_df.iloc[:-test_days]
    # )
    # prediction = autots.predict()
    # y_pred_autots = prediction.forecast[target].values
    # results.append(collect_metrics("AUTOTS", y_test.values, y_pred_autots))
    # print_metrics("AUTOTS", y_test.values, y_pred_autots)

    # Model Comparison Table
    results_df = pd.DataFrame(results)
    print("\n=== MODEL COMPARISON ===")
    print(results_df.sort_values("RMSE"))

    # collect all model results
    model_preds = {
        "BASELINE": y_pred_base.values,
        "LIGHTGBM": y_pred_lgb,
        "XGBOOST": y_pred_xgb,
        "ARIMA": y_pred_arima.values,
        "PROPHET": y_pred_prophet,
        "LSTM": y_pred_lstm
        #"AUTOTS": y_pred_autots
    }
    print("\n=== Direction Accuracy Table ===")
    print(direction_summary_table(
        model_preds,
        y_test.values
    ))

    # plot_confusion_multi(
    #     model_preds,
    #     y_test.values
    # )

    # plot_model_predictions(
    #     y_test.values,
    #     model_preds,
    #     title=f"{target} — Model Forecast Comparison"
    # )

    # ---------- SELECT BEST MODEL ----------
    best_row = results_df.sort_values("RMSE").iloc[0]
    best_model_name = best_row["model"]
    print("\nBest model:", best_model_name)

    # dict for mapping
    model_objects = {
        "BASELINE": ("sklearn", baseline), 
        "LIGHTGBM": ("sklearn", lgb),
        "XGBOOST": ("sklearn", xgb),
        "ARIMA": ("pickle", arima),
        "PROPHET": ("pickle", prop),
        "LSTM": ("keras", lstm)
        # "AUTOTS": ("pickle", autots)
    }

    # save best model
    if best_model_name in model_objects:
        mtype, mobj = model_objects[best_model_name]

        if mtype == "sklearn":
            save_sklearn_model(mobj, target, best_model_name)
        elif mtype == "pickle":
            save_pickle_model(mobj, target, best_model_name)
        elif mtype == "keras":
            save_keras_model(mobj, target, best_model_name)

        save_metadata(
            target,
            best_model_name,
            best_row.to_dict(),
            feat_cols
        )
    else:
        print("Best model not savable type — skipped")

    # # test loading model
    # loaded = load_model(target, best_model_name)
    # print(type(loaded))

def main():
    DATA_PATH = "data/Foreign_Exchange_Rates.csv"
    TARGET = "THB/USD"
    TEST_DAYS = 60

    # ---- STEP 2.1 ----
    df = load_and_prepare_data(DATA_PATH)

    # # quick validation target exists
    # if TARGET not in df.columns:
    #     raise ValueError(f"{TARGET} not found in columns")

    currency_cols = [
        c for c in df.columns
        if c != "Date"
    ]

    print("\nCurrencies:", currency_cols)

    for col in currency_cols:
        try:
            train_one_curr(df, col, TEST_DAYS)
        except Exception as e:
            print("FAILED:", col, e)

if __name__ == "__main__":
    main()


# import joblib
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_absolute_error
# from sklearn.model_selection import train_test_split

# from load_data import load_forex_data
# from preprocess import create_features


# TARGET = "USD/EUR"
# MODEL_PATH = "models/usd_eur_rf.pkl"


# def train():
#     df = load_forex_data("data/Foreign_Exchange_Rates.xls")
#     df = create_features(df, TARGET)

#     X = df.drop(columns=["Date", TARGET])
#     y = df[TARGET]

#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, shuffle=False, test_size=0.2
#     )

#     model = RandomForestRegressor(
#         n_estimators=200,
#         max_depth=10,
#         random_state=42
#     )

#     model.fit(X_train, y_train)
#     preds = model.predict(X_test)

#     mae = mean_absolute_error(y_test, preds)
#     print(f"MAE: {mae:.6f}")

#     joblib.dump(model, MODEL_PATH)
#     print(f"Model saved → {MODEL_PATH}")


# if __name__ == "__main__":
#     train()