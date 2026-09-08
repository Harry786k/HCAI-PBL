import csv
import random
import uuid
import pandas as pd

from functools import lru_cache
from datetime import datetime
from django.conf import settings
from django.shortcuts import render, redirect

from .features import extract_features


DATASET_PATH = settings.BASE_DIR / "pbl" / "movie_metadata.csv"
RESULTS_PATH = settings.BASE_DIR / "project4" / "study_results.csv"


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


def save_result(participant_id, task, response):
    new_file = not RESULTS_PATH.exists()

    with open(RESULTS_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if new_file:
            writer.writerow([
                "participant_id",
                "task",
                "response",
                "timestamp"
            ])

        writer.writerow([
            participant_id,
            task,
            response,
            datetime.now().isoformat(timespec="seconds")
        ])


def index(request):
    return render(request, "project4/index.html")


def design1_view(request):
    if request.method == "POST":
        participant_id = request.POST["participant_id"]
        movie1 = request.POST["movie1"]
        movie2 = request.POST["movie2"]
        selected = request.POST["preference"]

        save_result(
            participant_id,
            "pairwise",
            f"{movie1} VS {movie2} -> {selected}"
        )

        return redirect(
            f"/project4/design2/?pid={participant_id}"
        )

    participant_id = request.GET.get(
        "pid",
        str(uuid.uuid4())[:8]
    )

    df = pd.read_csv(DATASET_PATH).dropna(
        subset=[
            "movie_title",
            "title_year",
            "imdb_score",
            "genres"
        ]
    )

    movies = df.sample(2)

    return render(
        request,
        "project4/design1.html",
        {
            "movie1": movies.iloc[0],
            "movie2": movies.iloc[1],
            "participant_id": participant_id,
        }
    )


def design2_view(request):
    participant_id = (
        request.POST.get("participant_id")
        or request.GET.get("pid")
    )

    if request.method == "POST":
        ranking = request.POST.get("ranked_order", "")

        if ranking:
            save_result(
                participant_id,
                "ranking",
                ranking.replace("||", " > ")
            )

        return redirect("/project4/complete/")

    return render(
        request,
        "project4/design2.html",
        {
            "movies": random.sample(get_movies(), 10),
            "participant_id": participant_id,
        }
    )


def complete_view(request):
    return render(request, "project4/complete.html")