from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(exist_ok=True)

MOVIES_FILE = DATA_DIR / "tmdb_5000_movies.csv"
CREDITS_FILE = DATA_DIR / "tmdb_5000_credits.csv"

MODEL_FILE = MODEL_DIR / "model.pkl"
SIMILARITY_FILE = MODEL_DIR / "similarity.pkl"