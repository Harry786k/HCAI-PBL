import random
import pandas as pd

from functools import lru_cache
from django.conf import settings
from django.shortcuts import render

from .features import extract_features


DATASET_PATH = settings.BASE_DIR / "pbl" / "movie_metadata.csv"


@lru_cache(maxsize=1)
def get_movies():
    df = extract_features(str(DATASET_PATH))

    return (
        df["movie_title"]
        .dropna()
        .astype(str)
        .str.strip()
        .drop_duplicates()
        .tolist()
    )


def index(request):
    return render(request, "project4/index.html")


def design1_view(request):
    message = ""

    if request.method == "POST":
        selected = request.POST.get("preference")

        if selected:
            message = f"You selected: {selected}"

    df = pd.read_csv(DATASET_PATH).dropna(
        subset=["movie_title", "title_year", "imdb_score", "genres"]
    )

    df["movie_title"] = df["movie_title"].astype(str).str.strip()
    df = df.drop_duplicates(subset=["movie_title"])

    movies = df.sample(2)

    return render(
        request,
        "project4/design1.html",
        {
            "movie1": movies.iloc[0],
            "movie2": movies.iloc[1],
            "message": message,
        }
    )


def design2_view(request):
    message = ""

    if request.method == "POST":
        ranking = request.POST.get("ranked_order", "")

        if ranking:
            message = "Ranking submitted successfully."

    return render(
        request,
        "project4/design2.html",
        {
            "movies": random.sample(get_movies(), 10),
            "message": message,
        }
    )