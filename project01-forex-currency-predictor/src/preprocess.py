import pandas as pd
import numpy as np

def clean_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ensure Date to be an temporary index for doing interpolate
    df = df.sort_values("Date")

    value_cols = df.columns.drop("Date")

    # interpolate missing value (using before and after day)
    df[value_cols] = df[value_cols].interpolate(
        method="linear",
        limit_direction="both"
    )

    return df


def create_features(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    if target_col not in df.columns:
        raise ValueError(
            f"target_col '{target_col}' not found. Available columns: {df.columns.tolist()}"
        )

    df = df.copy()

    # ---- time features ----
    df['dayofweek'] = df['Date'].dt.dayofweek
    df['month'] = df['Date'].dt.month
    df['dayofmonth'] = df['Date'].dt.day

    # ---- lag features ----
    df['lag_1'] = df[target_col].shift(1)
    df['lag_7'] = df[target_col].shift(7)
    df['lag_14'] = df[target_col].shift(14)

    # ---- rolling features ----
    df['rolling_mean_7'] = df[target_col].rolling(7).mean()
    df['rolling_std_7'] = df[target_col].rolling(7).std()

    return df

def make_ml_features(df: pd.DataFrame, target: str) -> pd.DataFrame:

    df = df.copy()
    df = df.sort_values("Date")

    feature_cols = []

    # time features
    df["dow"] = df["Date"].dt.dayofweek
    df["month"] = df["Date"].dt.month

    # cyclic encoding
    df["dow_sin"] = np.sin(2*np.pi*df["dow"]/7)
    df["dow_cos"] = np.cos(2*np.pi*df["dow"]/7)
    
    feature_cols += ["dow", "month", "dow_sin", "dow_cos"]

    # lag features
    for lag in [1,2,3,5,7,14]:
        col = f"{target}_lag_{lag}"
        df[col] = df[target].shift(lag)
        feature_cols.append(col)

    # rolling
    df[f"{target}_roll_mean_7"] = df[target].rolling(7).mean()
    df[f"{target}_roll_std_7"]  = df[target].rolling(7).std()

    feature_cols += [
        f"{target}_roll_mean_7",
        f"{target}_roll_std_7"
    ]

    # return space features
    df[f"{target}_ret_1"] = df[target].pct_change(1)
    df[f"{target}_ret_3"] = df[target].pct_change(3)
    df[f"{target}_ret_7"] = df[target].pct_change(7)

    feature_cols += [
        f"{target}_ret_1",
        f"{target}_ret_3",
        f"{target}_ret_7"
    ]

    # momentum
    df[f"{target}_mom_5"] = df[target] - df[target].shift(5)
    feature_cols.append(f"{target}_mom_5")

    # cross-currency signals
    cross_candidates = [
        "EUR/USD",
        "JPY/USD",
        "GBP/USD",
        "CNY/USD"
    ]

    for c in cross_candidates:
        if c in df.columns and c != target:
            col = f"{c}_ret_1"
            df[col] = df[c].pct_change(1)
            feature_cols.append(col)

    return df, feature_cols

def make_lstm_dataset(series, window=20):

    X, y = [], []

    for i in range(window, len(series)):
        X.append(series[i-window:i])
        y.append(series[i])

    return np.array(X), np.array(y)


if __name__ == "__main__":
    from load_data import load_forex_data
    
    df = load_forex_data("data/Foreign_Exchange_Rates.csv")
    df = clean_missing_values(df)
    df_feat = create_features(df, target_col="EUR/USD")

    print(df_feat.head())
    print(df_feat.info())
    print(df_feat.isna().sum().sum())
    print(df.isna().sum().sum())

    ## plot graph ##
    import matplotlib.pyplot as plt
    import seaborn as sns

    # currency trend
    cols_to_plot = ["THB/USD", "EUR/USD", "JPY/USD"]
    df.plot(x="Date", y=cols_to_plot, figsize=(14,6))
    plt.title("Forex Rates Over Time")
    plt.show()

    # distribution plot
    df["THB/USD"].hist(bins=50)
    plt.title("THB/USD Distribution")
    plt.show()

    # return series
    returns = df.set_index("Date").pct_change()
    returns["THB/USD"].plot(figsize=(12,4))
    plt.title("THB/USD Returns")
    plt.show()

    # correlation matrix
    corr = df.drop(columns="Date").corr()
    plt.figure(figsize=(12,10))
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title("Currency Correlation Matrix")
    plt.show()