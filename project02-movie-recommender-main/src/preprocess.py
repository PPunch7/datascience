import ast

def parse_json_list(text):
    # Convert JSON-like string -> list of names
    try:
        data = ast.literal_eval(text)
        return [item["name"] for item in data]
    except:
        return []
    
def get_top_cast(text, limit=3):
    try:
        data = ast.literal_eval(text)
        return [item["name"] for item in data[:limit]]
    except:
        return []
    
def get_director(text):
    try:
        data = ast.literal_eval(text)
        for item in data:
            if item["job"] == "Director":
                return [item["name"]]
    except:
        pass

    return []

def clean_tokens(token_list):
    return [token.replace(" ", "") for token in token_list]

def preprocess_movies(df):
    df = df.copy()

    df["genres"] = df["genres"].apply(parse_json_list)
    df["keywords"] = df["keywords"].apply(parse_json_list)
    df["cast"] = df["cast"].apply(get_top_cast)
    df["crew"] = df["crew"].apply(get_director)

    # overview -> word list
    df["overview"] = df["overview"].fillna("").apply(lambda x: x.split())

    # remove spaces
    for col in ["genres", "keywords", "cast", "crew"]:
        df[col] = df[col].apply(clean_tokens)

    return df