"""
recommend.py
------------
Couples the two halves of TasteTrace:

  HMM -> where is this user in taste-space NOW, and where are they likely to go next?
  BN  -> given a taste state, how plausible is a movie with attributes (genre, era, popularity)?

score(movie | user) = sum over next-states s of  P(next=s | current state) * P_BN_s(movie attributes)

Explanations are generated from the same quantities: the attribute whose probability
in the user's state is most elevated relative to the population is cited as the "reason".
"""
import itertools
import numpy as np

from bn_model import ATTRS


def _joint_prob(bn, assignment):
    """P(x) = prod_i P(x_i | parents(x_i)), read straight from the BN's CPDs."""
    p = 1.0
    for node in bn.nodes():
        cpd = bn.get_cpds(node)
        evidence = {parent: assignment[parent] for parent in cpd.variables[1:]}
        try:
            p *= cpd.get_value(**{node: assignment[node]}, **evidence)
        except Exception:
            p *= 1e-6  # attribute value never seen in this state's training data
    return float(p)


class TasteTraceRecommender:
    def __init__(self, hmm_model, bns, tables, movies, train_sequences, state_paths):
        self.hmm = hmm_model
        self.bns = bns
        self.movies = movies
        self.train_sequences = train_sequences
        self.state_paths = state_paths
        self.watched = {u: {m for (m, _g, _r, _t) in seq} for u, seq in train_sequences.items()}

        # Precompute P_BN_s(genre, era, pop) lookup for every state
        self.joint = {}
        for s, bn in bns.items():
            domains = {a: list(bn.get_cpds(a).state_names[a]) for a in ATTRS}
            table = {}
            for combo in itertools.product(*(domains[a] for a in ATTRS)):
                assignment = dict(zip(ATTRS, combo))
                table[combo] = _joint_prob(bn, assignment)
            self.joint[s] = table

        # Marginals per state (for explanations) and globally
        import pandas as pd
        all_rows = pd.concat(tables.values())
        self.global_marg = {a: all_rows[a].value_counts(normalize=True).to_dict() for a in ATTRS}
        self.state_marg = {s: {a: df[a].value_counts(normalize=True).to_dict() for a in ATTRS} for s, df in tables.items()}

    # ---- user state ---------------------------------------------------------
    def current_state(self, uid):
        return int(self.state_paths[uid][-1])

    def next_state_dist(self, uid):
        return self.hmm.transmat_[self.current_state(uid)]

    # ---- scoring ------------------------------------------------------------
    def movie_key(self, movie_id):
        if not hasattr(self, "_keys"):
            self._keys = {m: (g, e, p) for m, g, e, p in zip(
                self.movies.index, self.movies["genre_cluster"], self.movies["era_bucket"], self.movies["popularity_tier"])}
        return self._keys[movie_id]

    def score_movie(self, uid, movie_id):
        key = self.movie_key(movie_id)
        dist = self.next_state_dist(uid)
        return float(sum(dist[s] * self.joint[s].get(key, 1e-9) for s in self.joint))

    def recommend(self, uid, k=10, candidates=None):
        if candidates is None:
            candidates = [m for m in self.movies.index if m not in self.watched[uid]]
        scored = sorted(((self.score_movie(uid, m), m) for m in candidates), reverse=True)
        return scored[:k]

    # ---- explanation --------------------------------------------------------
    def explain(self, uid, movie_id):
        s = self.current_state(uid)
        row = self.movies.loc[movie_id]
        values = {"genre_cluster": row["genre_cluster"], "era_bucket": row["era_bucket"], "popularity_tier": row["popularity_tier"]}
        lifts = {}
        for a in ATTRS:
            p_state = self.state_marg[s][a].get(values[a], 1e-6)
            p_glob = self.global_marg[a].get(values[a], 1e-6)
            lifts[a] = (p_state / p_glob, p_state)
        best_attr = max(lifts, key=lambda a: lifts[a][0])
        phrasing = {
            "genre_cluster": f"it is {values['genre_cluster']}, which makes up {lifts['genre_cluster'][1]:.0%} of what you watch in your current taste phase",
            "era_bucket": f"it is from the {values['era_bucket']} era, which is {lifts['era_bucket'][0]:.1f}x more common in your current taste phase than overall",
            "popularity_tier": f"it is a {values['popularity_tier']}-popularity title, {lifts['popularity_tier'][0]:.1f}x more typical of your current taste phase than overall",
        }
        secondary = [a for a in ATTRS if a != best_attr and lifts[a][0] > 1.05]
        text = f"Recommended because {phrasing[best_attr]}"
        if secondary:
            text += "; also matches your phase on " + " and ".join(a.replace("_", " ") for a in secondary)
        edges = list(self.bns[s].edges())
        return {"state": s, "text": text + ".", "lifts": {a: round(lifts[a][0], 2) for a in ATTRS}, "bn_edges": edges}
