"""
data_utils.py
-------------
Loads MovieLens-100K and builds the discretized, chronologically-ordered
representation TasteTrace needs:

  - each movie gets 3 discrete attributes: genre_cluster, era_bucket, popularity_tier
  - each user gets a chronological sequence of (movie_id, genre_cluster_code, rating, ts)

These 3 attributes are deliberately simple and human-readable, because they are
exactly what the Bayesian Network reasons over, and what the explanation text
is built from.
"""
import os
import pandas as pd
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "ml-100k")

GENRE_COLS = [
    "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery",
    "Romance", "Sci-Fi", "Thriller", "War", "Western",
]

# Genre clustering -> fewer, human-meaningful "taste" categories.
# This mirrors the base paper's idea of clustering rare/fine-grained genre
# flags into a small set of frequency-aware buckets, but the buckets here are
# chosen for interpretability rather than pure frequency.
GENRE_CLUSTER_MAP = {
    "Action": "Action/Adventure", "Adventure": "Action/Adventure", "Sci-Fi": "Action/Adventure",
    "War": "Action/Adventure", "Western": "Action/Adventure", "Fantasy": "Action/Adventure",
    "Comedy": "Family/Comedy", "Children's": "Family/Comedy", "Animation": "Family/Comedy", "Musical": "Family/Comedy",
    "Drama": "Drama/Romance", "Romance": "Drama/Romance",
    "Horror": "Dark/Thriller", "Thriller": "Dark/Thriller", "Crime": "Dark/Thriller",
    "Mystery": "Dark/Thriller", "Film-Noir": "Dark/Thriller",
    "Documentary": "Documentary",
    "unknown": "Drama/Romance",  # fallback bucket for the rare unflagged movies
}
GENRE_CLUSTERS = ["Action/Adventure", "Family/Comedy", "Drama/Romance", "Dark/Thriller", "Documentary"]
# Priority order used to break ties when several clusters are equally represented.
CLUSTER_PRIORITY = ["Dark/Thriller", "Action/Adventure", "Drama/Romance", "Family/Comedy", "Documentary"]


def _dominant_genre_cluster(row):
    """Pick the genre cluster with the most flags set for this movie; tie-break by priority."""
    counts = {c: 0 for c in GENRE_CLUSTERS}
    for g in GENRE_COLS:
        if row[g] == 1:
            counts[GENRE_CLUSTER_MAP[g]] += 1
    best = max(counts.items(), key=lambda kv: kv[1])
    if best[1] == 0:
        return "Drama/Romance"
    top_count = best[1]
    tied = [c for c, v in counts.items() if v == top_count]
    for c in CLUSTER_PRIORITY:
        if c in tied:
            return c
    return tied[0]


def _era_bucket(release_date):
    try:
        year = int(str(release_date).split("-")[-1])
    except (ValueError, IndexError):
        return "Unknown"
    if year < 1970:
        return "Pre-1970"
    if year < 1980:
        return "1970s"
    if year < 1990:
        return "1980s"
    if year < 1995:
        return "Early-1990s"
    return "Late-1990s"


def load_movies():
    """Returns a DataFrame indexed by movie_id with: title, genre_cluster, era_bucket."""
    cols = ["movie_id", "title", "release_date", "video_release_date", "imdb_url"] + GENRE_COLS
    movies = pd.read_csv(
        os.path.join(DATA_DIR, "u.item"), sep="|", names=cols, encoding="latin-1", index_col=False
    )
    movies["genre_cluster"] = movies.apply(_dominant_genre_cluster, axis=1)
    movies["era_bucket"] = movies["release_date"].apply(_era_bucket)
    return movies.set_index("movie_id")[["title", "genre_cluster", "era_bucket"]]


def load_ratings():
    """Returns a DataFrame of all ratings: user_id, movie_id, rating, timestamp."""
    ratings = pd.read_csv(
        os.path.join(DATA_DIR, "u.data"), sep="\t",
        names=["user_id", "movie_id", "rating", "timestamp"],
    )
    return ratings


def add_popularity_tier(ratings, movies):
    """Adds a popularity_tier column (Low/Medium/High) to `movies`, based on rating counts
    *within the training ratings only* to avoid leaking test-time popularity."""
    counts = ratings.groupby("movie_id").size()
    movies = movies.copy()
    movies["n_ratings"] = counts.reindex(movies.index).fillna(0).astype(int)
    # tertile split on movies that have at least 1 rating
    rated = movies[movies["n_ratings"] > 0]["n_ratings"]
    q1, q2 = rated.quantile([1/3, 2/3])

    def tier(n):
        if n == 0:
            return "Low"
        if n <= q1:
            return "Low"
        if n <= q2:
            return "Medium"
        return "High"

    movies["popularity_tier"] = movies["n_ratings"].apply(tier)
    return movies


def build_user_sequences(ratings, movies):
    """Returns dict: user_id -> list of (movie_id, genre_cluster, rating, timestamp),
    chronologically sorted."""
    merged = ratings.merge(movies[["genre_cluster"]], left_on="movie_id", right_index=True)
    merged = merged.sort_values(["user_id", "timestamp"])
    sequences = {}
    for uid, grp in merged.groupby("user_id"):
        sequences[uid] = list(zip(grp["movie_id"], grp["genre_cluster"], grp["rating"], grp["timestamp"]))
    return sequences


def train_test_split_leave_last_out(sequences):
    """For each user with >=5 interactions, holds out their LAST watched movie as the
    test item and returns (train_sequences, test_pairs) where test_pairs is
    {user_id: (held_out_movie_id, state_context_sequence)}."""
    train_sequences = {}
    test_pairs = {}
    for uid, seq in sequences.items():
        if len(seq) < 5:
            train_sequences[uid] = seq
            continue
        train_sequences[uid] = seq[:-1]
        test_pairs[uid] = seq[-1][0]  # held-out movie_id
    return train_sequences, test_pairs


if __name__ == "__main__":
    movies = load_movies()
    ratings = load_ratings()
    movies = add_popularity_tier(ratings, movies)
    print(movies.head(8))
    print("\nGenre cluster distribution:\n", movies["genre_cluster"].value_counts())
    print("\nEra bucket distribution:\n", movies["era_bucket"].value_counts())
    print("\nPopularity tier distribution:\n", movies["popularity_tier"].value_counts())
    seqs = build_user_sequences(ratings, movies)
    print(f"\nBuilt sequences for {len(seqs)} users. Example (user 1, first 5):")
    print(seqs[1][:5])
