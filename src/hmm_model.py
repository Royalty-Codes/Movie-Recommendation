"""
hmm_model.py
------------
Trains ONE shared Hidden Markov Model over every user's chronological
genre-cluster sequence (Baum-Welch, via hmmlearn's CategoricalHMM), then
decodes (Viterbi) each user's own sequence through it to recover their
personal path of "taste states" over time -- including their CURRENT state,
which is what the Bayesian Network half of TasteTrace conditions on.

Design note: unlike the base paper (which trains a separate small HMM per
user per feature, with rating-scores as hidden states), TasteTrace trains a
single HMM across all users so the hidden states are genuinely shared,
comparable "taste phases" (e.g. "Dark/Thriller phase"), and so that users
with short histories still benefit from the population's transition
statistics.
"""
import numpy as np
from hmmlearn import hmm

from data_utils import GENRE_CLUSTERS

GENRE_TO_CODE = {g: i for i, g in enumerate(GENRE_CLUSTERS)}
N_SYMBOLS = len(GENRE_CLUSTERS)


def sequences_to_hmm_input(train_sequences):
    """Flattens {user_id: [(movie_id, genre_cluster, rating, ts), ...]} into the
    (X, lengths, user_order) format hmmlearn needs for multi-sequence training."""
    codes = []
    lengths = []
    user_order = []
    for uid, seq in train_sequences.items():
        if len(seq) == 0:
            continue
        for (_mid, genre_cluster, _rating, _ts) in seq:
            codes.append(GENRE_TO_CODE[genre_cluster])
        lengths.append(len(seq))
        user_order.append(uid)
    X = np.array(codes).reshape(-1, 1)
    return X, lengths, user_order


def train_taste_hmm(train_sequences, n_states=4, random_state=42, n_iter=40, n_restarts=3):
    """Trains the shared CategoricalHMM via Baum-Welch. Since Baum-Welch only finds a
    local optimum, this restarts from several random initializations and keeps the one
    with the best training log-likelihood -- this matters here because a bad init can
    collapse two states into near-duplicates (same top genres), which defeats the point
    of having distinguishable taste phases."""
    X, lengths, _ = sequences_to_hmm_input(train_sequences)
    best_model, best_score = None, -np.inf
    for i in range(n_restarts):
        model = hmm.CategoricalHMM(
            n_components=n_states, n_features=N_SYMBOLS, n_iter=n_iter,
            random_state=random_state + i, init_params="ste",
        )
        model.fit(X, lengths)
        score = model.score(X, lengths)
        if score > best_score:
            best_model, best_score = model, score
    return best_model


def decode_user_states(model, seq):
    """Given one user's sequence, returns the Viterbi-decoded state at EACH time step.
    seq: list of (movie_id, genre_cluster, rating, ts)."""
    if len(seq) == 0:
        return []
    codes = np.array([GENRE_TO_CODE[g] for (_m, g, _r, _t) in seq]).reshape(-1, 1)
    _logprob, state_path = model.decode(codes, algorithm="viterbi")
    return list(state_path)


def decode_all_users(model, train_sequences):
    """Returns {user_id: state_path} for every user, where state_path[i] is the
    hidden taste-state the user was in when they watched their i-th movie."""
    return {uid: decode_user_states(model, seq) for uid, seq in train_sequences.items()}


def describe_states(model):
    """Human-readable summary of what each hidden state 'means' in genre terms --
    useful for sanity-checking the HMM and for writing slide/report commentary."""
    print(f"Learned {model.n_components}-state taste HMM over {N_SYMBOLS} genre clusters\n")
    for s in range(model.n_components):
        emission = model.emissionprob_[s]
        ranked = sorted(zip(GENRE_CLUSTERS, emission), key=lambda kv: -kv[1])
        top = ", ".join(f"{g} ({p:.2f})" for g, p in ranked[:3])
        print(f"  State {s}: mostly {top}")
    print("\nTransition matrix (rows=from, cols=to):")
    with np.printoptions(precision=2, suppress=True):
        print(model.transmat_)


if __name__ == "__main__":
    from data_utils import load_movies, load_ratings, add_popularity_tier, build_user_sequences, train_test_split_leave_last_out

    movies = load_movies()
    ratings = load_ratings()
    movies = add_popularity_tier(ratings, movies)
    sequences = build_user_sequences(ratings, movies)
    train_sequences, test_pairs = train_test_split_leave_last_out(sequences)

    model = train_taste_hmm(train_sequences, n_states=4)
    describe_states(model)

    state_paths = decode_all_users(model, train_sequences)
    uid = 1
    print(f"\nUser {uid}'s decoded taste-state path (first 10 steps): {state_paths[uid][:10]}")
    print(f"User {uid}'s CURRENT taste state: {state_paths[uid][-1]}")
