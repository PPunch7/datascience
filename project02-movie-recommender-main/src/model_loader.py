import pickle
from .config import MODEL_FILE, SIMILARITY_FILE

_movies = None
_similarity = None

def load_artifacts():
    global _movies, _similarity
    if _movies is None:
        print("Loading model artifacts...")
        _movies = pickle.load(open(MODEL_FILE, "rb"))
        _similarity = pickle.load(open(SIMILARITY_FILE, "rb"))

    return _movies, _similarity

def get_movie_list():
    movies, _ = load_artifacts()
    return sorted(movies["title"].tolist())
