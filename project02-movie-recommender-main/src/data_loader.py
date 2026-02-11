import pandas as pd
from .config import MOVIES_FILE, CREDITS_FILE

def load_data():
    movies = pd.read_csv(
        MOVIES_FILE, 
        engine="python", 
        sep="," ,
        quotechar='"'
    )
    credits = pd.read_csv(
        CREDITS_FILE, 
        engine="python", 
        sep="," ,
        quotechar='"'
    )

    print("Credits shapte:", credits.shape)
    print("Credits cols:", credits.columns)

    movies = movies.merge(credits, on="title")
    print("Movies shape:", movies.shape)
    return movies

if __name__ == "__main__":
    df = load_data()
    print(df.head())
    print(df.shape)
    print(df.info())
    print(df.describe())