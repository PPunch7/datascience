from sklearn.metrics.pairwise import cosine_similarity

def compute_similarity(vectors):
    similarity = cosine_similarity(vectors)
    return similarity