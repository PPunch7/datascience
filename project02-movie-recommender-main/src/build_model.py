import pickle

from .data_loader import load_data
from .preprocess import preprocess_movies
from .feature_builder import build_feature_matrix
from .config import MODEL_FILE, SIMILARITY_FILE

def main():
    print("Loading data...")
    df = load_data()

    print("Preprocessing...")
    df = preprocess_movies(df)

    print("Building vectors...")
    movies, vectors, similarity = build_feature_matrix(df)

    print("Saving model files...")
    MODEL_FILE.parent.mkdir(exist_ok=True)

    pickle.dump(movies, open(MODEL_FILE, "wb"))
    pickle.dump(similarity, open(SIMILARITY_FILE, "wb"))

    print("Saving done")

if __name__ == "__main__":
    main()
