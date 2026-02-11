from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

def stem_tokens(tokens):
    return [ps.stem(t) for t in tokens]

def build_tags(df):
    df = df.copy()

    # combine features
    df["tags"] = (
        df["overview"]
        + df["genres"]
        + df["keywords"]
        + df["cast"]
        + df["crew"]
    )

    # stemming
    df["tags"] = df["tags"].apply(stem_tokens)

    # join to string
    df["tags"] = df["tags"].apply(lambda x: " ".join(x))

    return df[["id", "title", "tags"]]

def stem_text(text):
    return " ".join(ps.stem(w) for w in text.split())

def build_feature_matrix(df):
    df = df.copy()

    df["tags"] = (
        df["overview"].apply(lambda x: " ".join(x))
        + " "
        + df["genres"].apply(" ".join)
        + " "
        + df["keywords"].apply(" ".join)
        + " "
        + df["cast"].apply(" ".join)
        + " "
        + df["crew"].apply(" ".join)
    )

    df["tags"] = df["tags"].apply(stem_text)
    movies = df[["id", "title", "tags"]]

    vectorizer = CountVectorizer(
        max_features=5000,
        stop_words="english"
    )

    vectors = vectorizer.fit_transform(movies["tags"]).toarray()
    similarity = cosine_similarity(vectors)

    return movies, vectors, similarity