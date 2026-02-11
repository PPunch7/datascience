from sklearn.feature_extraction.text import CountVectorizer

def build_vectors(tags_series, max_features=5000):
    vectorizer = CountVectorizer(
        max_features=max_features,
        stop_words="english"
    )

    vectors = vectorizer.fit_transform(tags_series).toarray()

    return vectors, vectorizer