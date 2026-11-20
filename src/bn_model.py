"""
bn_model.py
-----------
For each hidden taste-state discovered by the HMM, learns a small Bayesian
Network over three human-readable movie attributes:

    genre_cluster   (Action/Adventure, Family/Comedy, Drama/Romance, Dark/Thriller, Documentary)
    era_bucket      (Pre-1970, 1970s, 1980s, Early-1990s, Late-1990s)
    popularity_tier (Low, Medium, High)

Structure is learned (not assumed) via Hill-Climb search + BIC score, so the
causal story ("genre predicts era" vs "era predicts genre" vs independent) is
whatever the data in that state actually supports. Parameters are then fit
with a Bayesian (Dirichlet-prior) estimator so no probability is ever
exactly zero -- which matters because we'll multiply these probabilities
together when scoring candidate movies.
"""
import pandas as pd
from pgmpy.estimators import HillClimbSearch, BayesianEstimator
from pgmpy.models import DiscreteBayesianNetwork

ATTRS = ["genre_cluster", "era_bucket", "popularity_tier"]


def build_state_training_tables(train_sequences, state_paths, movies):
    """Returns {state: DataFrame[genre_cluster, era_bucket, popularity_tier]} --
    one row per (user, watched-movie) event, grouped by the state the user was
    decoded to be in at that moment."""
    attr_lookup = movies[ATTRS].to_dict("index")
    rows_by_state = {}
    for uid, seq in train_sequences.items():
        path = state_paths[uid]
        for (movie_id, _g, _r, _ts), state in zip(seq, path):
            rows_by_state.setdefault(int(state), []).append(attr_lookup[movie_id])
    return {s: pd.DataFrame(rows) for s, rows in rows_by_state.items()}


def learn_bn_for_state(df, equivalent_sample_size=10):
    """Learns structure (HillClimb + BIC) then fits CPDs (Bayesian estimator) for one state."""
    if len(df) < 20:
        # Too little data for this state to learn a reliable structure -- fall back
        # to a structure-free (fully independent) network, still fit with smoothing.
        model = DiscreteBayesianNetwork()
        model.add_nodes_from(ATTRS)
    else:
        hc = HillClimbSearch(df)
        best_structure = hc.estimate(scoring_method="bic-d", max_indegree=2, show_progress=False)
        model = DiscreteBayesianNetwork(best_structure.edges())
        model.add_nodes_from(ATTRS)  # in case HillClimb drops an unconnected node
    estimator = BayesianEstimator(model, df)
    cpds = estimator.get_parameters(prior_type="BDeu", equivalent_sample_size=equivalent_sample_size)
    model.add_cpds(*cpds)
    return model


def learn_all_state_bns(train_sequences, state_paths, movies):
    tables = build_state_training_tables(train_sequences, state_paths, movies)
    bns = {}
    for state, df in tables.items():
        bns[state] = learn_bn_for_state(df)
    return bns, tables


def describe_bns(bns, tables):
    for state in sorted(bns):
        edges = list(bns[state].edges())
        n = len(tables[state])
        print(f"State {state} (n={n} watch-events): learned edges = {edges if edges else '(none -- independent attributes)'}")


if __name__ == "__main__":
    from data_utils import load_movies, load_ratings, add_popularity_tier, build_user_sequences, train_test_split_leave_last_out
    from hmm_model import train_taste_hmm, decode_all_users, describe_states

    movies = load_movies()
    ratings = load_ratings()
    movies = add_popularity_tier(ratings, movies)
    sequences = build_user_sequences(ratings, movies)
    train_sequences, test_pairs = train_test_split_leave_last_out(sequences)

    model = train_taste_hmm(train_sequences, n_states=4)
    describe_states(model)
    state_paths = decode_all_users(model, train_sequences)

    bns, tables = learn_all_state_bns(train_sequences, state_paths, movies)
    print()
    describe_bns(bns, tables)

    print("\nExample CPD -- genre_cluster in state 0:")
    print(bns[0].get_cpds("genre_cluster"))
