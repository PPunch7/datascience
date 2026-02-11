import pandas as pd
from pathlib import Path


def load_forex_data(file_path: str) -> pd.DataFrame:
    # string → Path object
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if file_path.suffix.lower() == ".csv":
        df = pd.read_csv(file_path)
    elif file_path.suffix.lower() in [".xls", ".xlsx"]:
        df = pd.read_excel(file_path, engine="openpyxl")
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")

    # clean columns
    df.columns = df.columns.str.strip()
    # print(df.columns)

    # drop unnecessary columns
    df = df.drop(columns=["Unnamed: 0", "Unnamed: 24"], errors="ignore")

    # change column names
    df = df.rename(columns={
        "Time Serie": "Date"
        , "AUSTRALIA - AUSTRALIAN DOLLAR/US$": "AUD/USD"
        , "EURO AREA - EURO/US$": "EUR/USD"
        , "NEW ZEALAND - NEW ZELAND DOLLAR/US$": "NZD/USD"
        , "UNITED KINGDOM - UNITED KINGDOM POUND/US$": "GBP/USD"
        , "BRAZIL - REAL/US$": "BRL/USD"
        , "CANADA - CANADIAN DOLLAR/US$": "CAD/USD"
        , "CHINA - YUAN/US$": "CNY/USD"
        , "HONG KONG - HONG KONG DOLLAR/US$": "HKD/USD"
        , "INDIA - INDIAN RUPEE/US$": "INR/USD"
        , "KOREA - WON/US$": "KRW/USD"
        , "MEXICO - MEXICAN PESO/US$": "MXN/USD"
        , "SOUTH AFRICA - RAND/US$": "ZAR/USD"
        , "SINGAPORE - SINGAPORE DOLLAR/US$": "SGD/USD"
        , "DENMARK - DANISH KRONE/US$": "DKK/USD"
        , "JAPAN - YEN/US$": "JPY/USD"
        , "MALAYSIA - RINGGIT/US$": "MYR/USD"
        , "NORWAY - NORWEGIAN KRONE/US$": "NOK/USD"
        , "SWEDEN - KRONA/US$": "SEK/USD"
        , "SRI LANKA - SRI LANKAN RUPEE/US$": "LKR/USD"
        , "SWITZERLAND - FRANC/US$": "CHF/USD"
        , "TAIWAN - NEW TAIWAN DOLLAR/US$": "TWD/USD"
        , "THAILAND - BAHT/US$": "THB/USD"})
    
    # convert Date column
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
    df = df.sort_values("Date").reset_index(drop=True)

    # replace ND → NaN and change to be a numeric number
    value_cols = df.columns.drop("Date")
    df[value_cols] = (
        df[value_cols]
            .replace("ND", None)
            .apply(pd.to_numeric, errors="coerce")
    )

    return df


if __name__ == "__main__":
    df = load_forex_data("data/Foreign_Exchange_Rates.csv")
    print(df.head())

    print(df.shape)
    print(df.info())
    print(df.describe())

    print(df["Date"].min(), df["Date"].max())

    ## check missing data
    import matplotlib.pyplot as plt
    import seaborn as sns
    df.isna().sum().sort_values(ascending=False)
    plt.figure(figsize=(12,6))
    sns.heatmap(df.isna(), cbar=False)
    plt.title("Missing Value Map")
    plt.show()