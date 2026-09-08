import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def extract_features(filepath):
    df = pd.read_csv(filepath)[
        ["movie_title", "genres", "imdb_score", "title_year"]
    ].dropna().copy()

    genres = df["genres"].str.get_dummies(sep="|")

    scaler = MinMaxScaler()
    df[["imdb_score_scaled", "title_year_scaled"]] = scaler.fit_transform(
        df[["imdb_score", "title_year"]]
    )

    return pd.concat([
        df[["movie_title", "imdb_score_scaled", "title_year_scaled"]],
        genres
    ], axis=1)