from flask import Blueprint, render_template, request
from src.recommender_engine import recommend_movie
from src.model_loader import get_movie_list

main_routes = Blueprint("main", __name__)

def fake_posters(titles):
    # placeholder images
    return [
        f"https://dummyimage.com/300x450/000/fff&text={t.replace(' ', '+')}"
        for t in titles
    ]

@main_routes.route("/", methods=["GET"])
def home():
    movie_list = get_movie_list()

    return render_template(
        "index.html",
        movie_list=movie_list,
        recommended_movie_titles=[],
        recommended_movie_posters=[]
    )

@main_routes.route("/recommend", methods=["POST"])
def recommend():
    selected = request.form.get("selected_movie")
    rec_df = recommend_movie(selected, top_k=8)
    titles = rec_df["title"].tolist()
    posters = fake_posters(titles)

    print(posters[:3])

    movie_list = get_movie_list()

    return render_template(
        "index.html",
        movie_list=movie_list,
        recommended_movie_titles=titles,
        recommended_movie_posters=posters
    )
