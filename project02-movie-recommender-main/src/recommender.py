import numpy as np

def recommend(title, movies_df, similarity, top_k=10):
    # find movie index
    matches = movies_df[movies_df["title"] == title]

    if len(matches) == 0:
        raise ValueError(f"Movie '{title}' not found")

    idx = matches.index[0]

    # similarity scores
    scores = list(enumerate(similarity[idx]))

    # DESC sorting
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    # skip first item
    top = scores[1: top_k + 1]

    results = movies_df.iloc[[i[0] for i in top]]
    return results[["title"]]
