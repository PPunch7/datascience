from .model_loader import load_artifacts
from .recommender import recommend

def recommend_movie(title, top_k=10):
    movies, similarity = load_artifacts()

    return recommend(title, movies, similarity, top_k)
