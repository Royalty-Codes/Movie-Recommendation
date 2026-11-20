"""
main.py -- end-to-end TasteTrace run on MovieLens-100K.

Evaluation protocol (standard leave-one-out with sampled negatives):
  for each user, hold out their LAST watched movie; rank it against 99 random
  unwatched movies; report HitRate@10 and NDCG@10.
"""
import os, sys, warnings, random
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
import pandas as pd

from data_utils import (load_movies, load_ratings, add_popularity_tier,
                        build_user_sequences, train_test_split_leave_last_out, GENRE_CLUSTERS)
from hmm_model import train_taste_hmm, decode_all_users, describe_states, GENRE_TO_CODE
from bn_model import learn_all_state_bns, describe_bns
from recommend import TasteTraceRecommender

N_STATES = 4
N_NEG = 99
K = 10
SEED = 7


def main():
    rng = random.Random(SEED)
    movies = load_movies()
    ratings = load_ratings()

    # 1. chronological sequences + leave-last-out split
    seqs = build_user_sequences(ratings, movies)
    train_seqs, test_pairs = train_test_split_leave_last_out(seqs)

    # 2. popularity tier from TRAIN ratings only (no test leakage)
    train_ratings = pd.DataFrame(
        [(u, m, r, t) for u, seq in train_seqs.items() for (m, _g, r, t) in seq],
        columns=["user_id", "movie_id", "rating", "timestamp"])
    movies = add_popularity_tier(train_ratings, movies)

    # 3. HMM taste states
    cache = os.path.join(os.path.dirname(__file__), "hmm_cache.pkl")
    if os.path.exists(cache):
        import pickle
        hmm_model = pickle.load(open(cache, "rb"))
    else:
        import pickle
        hmm_model = train_taste_hmm(train_seqs, n_states=N_STATES)
        pickle.dump(hmm_model, open(cache, "wb"))
    print("=== HMM taste states ===")
    describe_states(hmm_model)
    state_paths = decode_all_users(hmm_model, train_seqs)

    # 4. BN per state
    bns, tables = learn_all_state_bns(train_seqs, state_paths, movies)
    print("\n=== Learned Bayesian-network structure per state ===")
    describe_bns(bns, tables)

    rec = TasteTraceRecommender(hmm_model, bns, tables, movies, train_seqs, state_paths)

    # 5. baselines
    pop = movies["n_ratings"].to_dict()
    emis = hmm_model.emissionprob_

    gcode = {m: GENRE_TO_CODE[g] for m, g in zip(movies.index, movies["genre_cluster"])}

    def hmm_only_score(uid, m):
        dist = rec.next_state_dist(uid)
        g = gcode[m]
        return float(sum(dist[s] * emis[s][g] for s in range(N_STATES)))

    # --- personalization variants: BN prior x user's own recent attribute affinity (+ popularity) ---
    from collections import Counter
    keyof = {m: rec.movie_key(m) for m in movies.index}
    T_RECENT = 20
    aff = {}
    for u, seq in train_seqs.items():
        recent = [keyof[m] for (m, _g, _r, _t) in seq[-T_RECENT:]]
        cg, ce = Counter(k[0] for k in recent), Counter(k[1] for k in recent)
        aff[u] = (cg, ce, len(recent))

    def affinity(uid, m):
        cg, ce, n = aff[uid]
        k = keyof[m]
        return ((cg[k[0]] + 1) / (n + 5)) * ((ce[k[1]] + 1) / (n + 5))

    def tt_affinity(uid, m):
        return np.log(rec.score_movie(uid, m)) + np.log(affinity(uid, m))

    def tt_affinity_pop(uid, m):
        return tt_affinity(uid, m) + 0.5 * np.log1p(pop[m])

    methods = {
        "Popularity": lambda u, m: pop[m],
        "HMM-only (genre emission)": hmm_only_score,
        "TasteTrace (HMM + BN)": rec.score_movie,
        "TasteTrace + user affinity": tt_affinity,
        "TasteTrace + affinity + popularity": tt_affinity_pop,
    }
    hits = {k: [] for k in methods}
    ndcgs = {k: [] for k in methods}

    all_movies = list(movies.index)
    for uid, target in test_pairs.items():
        unwatched = [m for m in all_movies if m not in rec.watched[uid] and m != target]
        cands = rng.sample(unwatched, N_NEG) + [target]
        for name, fn in methods.items():
            # tiny random tiebreak so ties don't favor the test item's list position
            scored = sorted(cands, key=lambda m: (fn(uid, m), rng.random()), reverse=True)
            rank = scored.index(target)
            hits[name].append(1.0 if rank < K else 0.0)
            ndcgs[name].append(1.0 / np.log2(rank + 2) if rank < K else 0.0)

    print(f"\n=== Evaluation: {len(test_pairs)} users, 1 held-out item vs {N_NEG} sampled negatives ===")
    print(f"{'Method':32s} {'HR@10':>8s} {'NDCG@10':>9s}")
    for name in methods:
        print(f"{name:32s} {np.mean(hits[name]):8.3f} {np.mean(ndcgs[name]):9.3f}")

    # 6. explained recommendations
    print("\n=== Example explained recommendations ===")
    for uid in [1, 42, 200]:
        s = rec.current_state(uid)
        print(f"\nUser {uid} | current taste state: {s} | next-state dist: {np.round(rec.next_state_dist(uid), 2)}")
        for score, m in rec.recommend(uid, k=3):
            ex = rec.explain(uid, m)
            print(f"  * {movies.loc[m, 'title']}  [{rec.movie_key(m)}]")
            print(f"      {ex['text']}")


if __name__ == "__main__":
    main()
