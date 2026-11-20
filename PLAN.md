# TasteTrace — Complete Implementation Plan (for the coding agent)

> **Audience:** an autonomous coding agent (Antigravity) implementing this project end-to-end.
> **Owner:** Roydon. **Deadline:** 31 Oct 2026. **Plan written:** 10 Oct 2026.
> **Project title:** *TasteTrace: A Temporal-Causal Hybrid Model for Explainable Movie Recommendations Using Hidden Markov Models and Bayesian Networks*
> **Dataset:** MovieLens-100K (primary), MovieLens-1M (stretch).

Work **phase by phase**. Every phase has *tasks*, *deliverables* and *acceptance criteria*. Do not move on until the acceptance criteria pass. **Never invent or hand-edit numbers** — every number in the final comparison must be produced by a script and written to `results/`.

---

## 0. Project in one page

**Goal.** Build an explainable movie recommender that combines
(a) a Hidden Markov Model tracking a user's evolving *taste phase*, and
(b) per-phase Bayesian Networks giving human-readable reasons,
and show — under a **fair held-out protocol** — that it **outperforms the base paper** and is **competitive with strong classical baselines**, while being the only one that explains each recommendation.

**Base paper (the thing we must beat).**
Keudjeu, Kengni Ngangmo, Iloga, Bouetou Bouetou, *"Hybrid Movie Recommendation Based On Recent User Sequential Preferences And Sequential Profiles Using HMMs"*, IEEE Access 2025, DOI 10.1109/ACCESS.2025.3630634. The owner has the PDF; their authors' code is at `github.com/HmmsMovieRecoms/HMMs-MovieRecommendation` (consult it if reachable, otherwise follow §4).

**Critical finding that shapes the whole plan (verified from the paper's text):**
The paper's headline numbers (Precision 0.997, RMSE 0.0153, MAE 0.00036 on ML-100K) **cannot be matched or compared directly**, because:
1. It states the datasets were **not partitioned into train/validation/test**; evaluation is "on the entire datasets" (Sec. V-D).
2. Its rating predictor (Eq. 9) returns the user's own rating for movies already watched ("if u is a viewer of y, this approximation becomes a strict equality"), which drives RMSE/MAE to ~0.
3. Precision/recall are computed against `X_u` — the movies the user **already watched** — and the candidate set `R^F(u)` (movies watched by neighbors) is not required to exclude the user's own history.

Therefore "better than the base paper" is defined as: **reimplement their method faithfully, evaluate it AND ours under the same held-out protocol (§3), and win there.** Additionally, reproduce their *own* protocol to (a) sanity-check the reimplementation and (b) demonstrate that a trivial "recommend what the user already watched" baseline scores near-perfectly under it (research-gap evidence).

**Current state (starting code, in `tastetrace_project.zip`, 5-seed HR@10, sampled-99 protocol):**

| Method | HR@10 | NDCG@10 |
|---|---|---|
| Popularity | 0.407 | 0.228 |
| ItemKNN (cosine) | 0.508 | 0.294 |
| SVD-MF (k=32) | 0.617 | 0.370 |
| TasteTrace v1 (HMM+BN, 5 genre clusters) | 0.365 | 0.168 |
| TasteTrace v1 + affinity + pop | 0.461 | 0.245 |

v1 is far below SVD because its HMM/BN only see 5 coarse genre clusters (no collaborative signal) and many movies tie on score. **v2 (this plan) fixes this.**

**Success targets (report honestly if missed — do not fudge):**
- T1 (must): TasteTrace-v2 > Base-paper-reimplementation on HR@10 and NDCG@10, paired test p < 0.01.
- T2 (should): TasteTrace-v2 ≥ SVD-MF within its 95% CI, or better.
- T3 (must): every recommendation ships with an explanation, and explanation **faithfulness** (§8) is measured and clearly above a random-component control.
- T4 (must): a working demo (§10) and a one-glance comparison figure/table.

---

## 1. Environment & repo layout

**Python 3.10–3.12.** `requirements.txt`:
```
numpy pandas scipy scikit-learn hmmlearn pgmpy networkx matplotlib seaborn streamlit pyyaml tqdm statsmodels pytest graphviz
```

**Known library gotchas (already hit in v1 — respect them):**
- `pgmpy>=1.1`: use `from pgmpy.models import DiscreteBayesianNetwork` (not `BayesianNetwork`). `HillClimbSearch.estimate(scoring_method="bic-d", max_indegree=2, show_progress=False)` (string scores; `BicScore` no longer exists). Fit CPDs with `BayesianEstimator(model, df).get_parameters(prior_type="BDeu", equivalent_sample_size=10)` then `model.add_cpds(*cpds)` (`model.fit(..., prior_type=...)` is **not** supported).
- `hmmlearn`: use `CategoricalHMM` (not `MultinomialHMM`) with integer observations shaped `(n,1)` and a `lengths` list for multiple sequences. Baum-Welch is slow on ~100k observations (~45 s per restart): **cache fitted models to disk (pickle) keyed by config hash**.
- Never call `DataFrame.loc[...]` inside per-event/per-candidate loops (was a 60× slowdown). Convert to dicts/NumPy arrays first.
- Run long jobs unbuffered (`python -u`) with logs written to `results/logs/`; shell tool calls time out at ~300 s, so run long jobs in the background.
- Do **not** commit MovieLens data (GroupLens license forbids redistribution). `scripts/download_data.py` must fetch `https://files.grouplens.org/datasets/movielens/ml-100k.zip` (and `ml-1m.zip`) and unzip to `data/`.

**Target layout (refactor v1 `src/*.py` into a package):**
```
tastetrace/
  PLAN.md  README.md  config.yaml  requirements.txt
  scripts/  download_data.py  run_all.py  make_figures.py
  src/tastetrace/
    data.py          # loading, id maps, per-user chronological sequences
    splits.py        # leave-last-out: test = last, val = second-last, train = rest
    features.py      # item attributes (genre flags, primary genre, era, popularity tier, avg-rating tier)
    clusters.py      # SVD item embeddings -> KMeans "taste clusters" + cluster naming
    hmm.py           # shared CategoricalHMM over taste-cluster sequences (+ model selection)
    bn.py            # per-state Bayesian networks
    fusion.py        # learned linear ranker over explainable components
    explain.py       # explanation generation (templates + evidence)
    models/          # baselines.py  base_paper.py  tastetrace.py  (common interface)
    eval/            # protocol.py  metrics.py  stats.py  faithfulness.py
  app/streamlit_app.py      demo.py (CLI fallback)
  results/ (csv, json, logs)   figures/   tests/
```
**Common model interface** (all methods incl. baselines): `fit(train_data, val_data)` and `score(user_id, item_ids) -> np.ndarray`. The evaluator only calls `score`, so every method is judged identically.

`config.yaml` holds all hyperparameters, seeds, paths. No magic numbers in code.

---

## 2. Phase 0 — Setup & refactor (Oct 10–11)

Tasks
1. Create the layout above; port v1 code (`data_utils.py`, `hmm_model.py`, `bn_model.py`, `recommend.py`, `bench.py`) into the package; keep behaviour identical first.
2. `scripts/download_data.py` + `data.py` loaders for `u.data`, `u.item` (19 genre flags, `release_date`), `u.user` (age, gender, occupation, zip).
3. `splits.py`: per user with ≥5 interactions sort by timestamp (**break timestamp ties by original row order — ML-100K has many identical timestamps; keep it deterministic**). `test` = last item, `val` = second-last, `train` = the rest. Users with <5 interactions: all-train, excluded from evaluation.
4. Unit tests: no test/val item leaks into train; popularity tiers and any item statistics are computed from **train only**.

Acceptance: `pytest` green; `python scripts/run_all.py --smoke` reproduces v1 numbers within ±0.01.

---

## 3. Phase 1 — Evaluation harness (Oct 11–12) — **do this before any model work**

**Primary protocol (P1, "sampled-99"):** for each eligible user, candidates = held-out item + 99 random unseen items (unseen = not in train/val/test history). Same candidate list for **every method** per (user, seed). Seeds = 5 (candidate sampling).
**Secondary protocol (P2, "full-rank"):** rank the held-out item against **all** unseen items (≈1.6k). Reported because sampled metrics can mis-rank methods.
**Reference protocol (P0, "paper-protocol"):** replicate the base paper's evaluation: no split; top-n from the method's candidate set; precision/recall against the user's **own watched set**. Used only for §4 sanity check and the protocol-flaw demo.

Metrics: HR@10, NDCG@10, MRR, Recall@10 (P2), plus beyond-accuracy: catalogue coverage@10, mean item popularity of top-10 (novelty), per-user explanation count (§8).

Statistics (`eval/stats.py`): per-user paired differences; **paired bootstrap (10k resamples over users) 95% CI** and a **paired permutation test** for every "A beats B" claim; aggregate over seeds by averaging per-user metrics across seeds first, then bootstrapping users (do **not** pool seeds as independent samples — v1's CI was optimistic for that reason). Also report mean±std across model-init seeds (§6).

Deliverable: `results/main_comparison.csv` with mean, std, CI per (method, metric, protocol).
Acceptance: Popularity / ItemKNN / SVD-MF reproduce v1 within ±0.01 HR@10; harness unit-tested with a perfect oracle (HR=1) and random scorer (HR≈0.10).

---

## 4. Phase 1b — Baselines and faithful base-paper reimplementation (Oct 12–14)

### 4.1 Classical baselines (`models/baselines.py`)
Popularity (train counts) · ItemKNN-cosine · UserKNN-cosine · SVD-MF on the implicit train matrix (k tuned on val over {16,32,64,128}) · **BPR-MF** (implement in NumPy/PyTorch or use `implicit`; tune on val). Stretch: a small **SASRec-style** transformer (1–2 blocks) as the sequential reference. Tune every baseline on the **validation** item, never on test.

### 4.2 Base-paper reimplementation (`models/base_paper.py`) — "KeudjeuHMM"
Follow the paper exactly (section refs are to the paper). Document every assumption in `docs/base_paper_assumptions.md`.

**CBF half — user vectors `u_F` (Sec. IV-A/C).** For user *u* with watched sequence `X` = last `T_max` movies (chronological, train only), and movie features `F = {f1 release date, f2 vote average, f3 vote count, f4 genres}`:
1. For each feature `f_i` build a pseudo-Markov chain: **states = the user's rating of each movie, sampled into 5 bins** `[0,1], (1,2], (2,3], (3,4], >4`; **symbols = the movie's value of `f_i`**, discretized.
   - Release date bins (10): `≤1979, 1980–85, 86–90, 91–95, 96–2000, 2001–05, 06–10, 11–15, 16–20, ≥2021`.
   - Vote average bins (10, width 0.5 from 0 to 5, last `>4.5`); vote count bins (10, for 100K: `[0,25],[26,50],[51,75],[76,100],[101,125],[126,150],[151,200],[201,250],[251,300],≥301`). `vote average`/`vote count` are the mean/number of train ratings per movie.
   - Genres: count each genre's frequency, cluster genres into 4 clusters by frequency thresholds (paper's rule for 100K: `≥700→1, [400,700)→2, [100,400)→3, <100→4`; the paper doesn't say whether counts are over movies or ratings — pick **ratings**, ensure all 4 clusters are non-empty, and document it). A movie's genre symbol = integer of the 4-bit membership word (1..15).
2. Initialize the HMM from the chain by counting (Eq. 1–3) with `ε = 1.0`, then redistribute missing probability mass per row (Eq. 4), then **Baum-Welch with γ = 50 iterations**. Users with `|X| ≤ 1`: keep the initial HMM, no training.
3. Vector component (Eq. 5): `u_i[l] = Σ_j φ[s_j] · B[s_j, z_l]` where `φ` = stationary distribution approximated by rows of `A^r`, `r = 100`.
4. `u_F = concat(u_1..u_4)` (Eq. 6).

**CF half — movie vectors `x_G` (Sec. IV-B/C).** Mirror image: for movie *x* with its last `T̄_max` viewers (train only), states = each viewer's rating of *x* (5 bins), symbols = viewer features `G = {age (bins [0,11],[12,18],[19,30],[31,45],[46,60],[61,75],≥76), gender (2), occupation (clustered by frequency, 7 clusters), zip first character (clustered by frequency, 6 clusters)}`. Same Eq. 1–6.

**Recommendation (Sec. IV-D).**
- Distance: **Manhattan** (paper found it better than Euclidean). `D_F(u,v) = d(u_F, v_F)`; `D_G(x,y) = d(x_G, y_G)`.
- KNN: `K` nearest users (paper's best: 50 on 100K) → `V`.
- Candidate set `R^F(u)` = movies watched by neighbors in `V`.
- Importance (Eq. 11): `β(y,u) = ln( |V(y)| · Σ_{x∈X_u} α^u(x) / (D_G(y,x) + ϵ) )`, `ϵ = 1e-150`, `V(y)` = neighbors who watched `y`.
- Top-n by β. Cold start (Alg. 2/3) is **not** needed for the headline comparison; implement only if time permits.
- **Adapter for our fair protocol** (`score(u, items)`): `score = β(y,u)` if `|V(y)| > 0`, else `-inf` (random tie-break among `-inf`, as the method cannot recommend those). Also expose a variant `KeudjeuHMM+fallback` that scores `|V(y)|=0` items with the paper's predicted-score formula (Eq. 9) so it isn't unfairly penalized; **report both**, and use the stronger one as "the base paper" in the main table.
- Grid: `T_max = T̄_max ∈ {20,40,60,80,100}`, `K ∈ {10..50}`; pick best on **validation**.

**Efficiency:** train ~943×4 user HMMs and ~1682×4 movie HMMs, all tiny (≤100 observations, 5 states). Implement scaled forward-backward/Baum-Welch in vectorized NumPy (or `hmmlearn` with preset `startprob_/transmat_/emissionprob_` and `init_params=""`). Cache vectors to disk.

### 4.3 Reproduction sanity check (P0)
Run KeudjeuHMM under the **paper's own protocol** (no split, precision/recall vs. the user's watched set, top-10/20, `T_max` 20–100, `K=50`). Expect precision/NDCG in the paper's ballpark (P≈0.99 on 100K). If it deviates by >0.1, re-read §4.2 and the authors' code before proceeding; record the gap in `docs/reproduction_report.md`.
Then add the **"Recommend-your-own-history" trivial baseline** under P0 and show it also scores ≈1.0 → this is the protocol-flaw evidence for the presentation (a bar chart: *P0 scores* vs *P1 scores* for Popularity, SVD, KeudjeuHMM, trivial-history).

Acceptance (Phase 1b): all baselines + KeudjeuHMM produce rows in `results/main_comparison.csv`; reproduction report written; protocol-flaw figure generated.

---

## 5. Phase 2 — TasteTrace v2 core: collaborative taste clusters + HMM + BN (Oct 14–18)

**Why v1 was weak and what changes.** v1 used 5 coarse genre clusters as the only signal, so (i) no "who liked what" information, (ii) hundreds of movies share identical (genre, era, popularity) keys and tie. v2 injects collaborative signal into the *observations and attributes* of the HMM/BN (so it stays explainable) instead of bolting on a black box.

### 5.1 Taste clusters (`clusters.py`)
1. Build the implicit train matrix `R` (users × items, 1 if interacted). Truncated SVD (`k=32`, tune on val ∈ {16,32,64}); item embeddings `V` → L2-normalize.
2. KMeans on item embeddings → `C` **taste clusters**, `C ∈ {8,12,16,24}` (tune on val; `n_init≥10`, fixed seeds).
3. **Name every cluster** for explanations: top-3 genres by lift + 3 most representative titles (closest to centroid among popular items), e.g. *"Dark thrillers (Fargo, Se7en, …)"*. Store in `results/cluster_names.json`.
4. Items with no train interactions (cold items) → assign nearest cluster by genre-profile similarity (fallback: cluster of most similar genre vector).

### 5.2 Shared HMM over taste-cluster sequences (`hmm.py`)
- Observation per watched movie = its **taste cluster id**; sequences = each user's chronological **train** history. One shared `CategoricalHMM`, `n_states ∈ {3,4,5,6,8}`, ≥5 random restarts, keep best log-likelihood; cache.
- **Model selection on validation:** choose `n_states` (and `C`) maximizing **validation HR@10** (primary) with held-out log-likelihood as a sanity check. Report the selection curve (figure).
- Guard against degenerate states: if any state has <1% of decoded events, merge it or lower `n_states`; log this.
- Decode each user with Viterbi (and keep forward posteriors) → current state `s_u` and predictive next-state distribution `P(next=s) = Σ_k posterior(k)·A[k,s]` (use the **forward posterior at the last step**, not just the Viterbi state — smoother and better calibrated).
- Optional improvement (try, keep only if val improves): weight observations by rating (emit only items rated ≥4 as "liked" events) or use an input-output HMM with rating as input.

### 5.3 Per-state Bayesian networks (`bn.py`)
- Per state `s`, build a table from all (user, event) pairs decoded to `s` with attributes:
  `taste_cluster` (C levels), `primary_genre` (≈8 levels), `era` (5), `popularity_tier` (3), `avg_rating_tier` (3, from train means), optionally `user_age_bin` and `user_gender` as **root nodes** (so the BN can say *"people like you in this phase tend to…"*).
- Structure: `HillClimbSearch(...).estimate(scoring_method="bic-d", max_indegree=2)`; parameters: BDeu `equivalent_sample_size ∈ {5,10,20}` (val-tuned). States with <500 events: fall back to shrinkage toward the global BN (mix CPDs: `λ·state + (1-λ)·global`, `λ = n/(n+200)`).
- Item score under state `s`: `log P_BN_s(attrs(item))`. Precompute a lookup table over all attribute combinations that occur in the catalogue → O(1) scoring (vectorize to a `(n_states × n_items)` matrix `J`). Expected-phase prior score: `s_BN(u,i) = log Σ_s P(next=s|u)·J[s,i]`.
- Because attributes now include the taste cluster, ties largely disappear; verify the fraction of tied top-10 scores per user is <5%.

Deliverables: `results/hmm_selection.csv`, `figures/state_semantics.png` (each state → its dominant taste clusters), `figures/bn_structures.png` (learned DAG per state), cached models.
Acceptance: TasteTrace-core (just `s_BN`) beats v1-core by a clear margin on val HR@10; no degenerate states; BN lookup scoring for all users×items runs in <30 s.

---

## 6. Phase 3 — Explainable fusion ranker + explanations (Oct 19–22)

### 6.1 Components (each is human-interpretable; compute for every (user, item), standardize per user across the candidate pool)
| id | component | meaning (used in explanation text) |
|---|---|---|
| `c_phase` | `s_BN(u,i)` | fits what people in your **current/next taste phase** watch |
| `c_recent` | log-affinity of item's cluster/genre/era to the user's last 20 events (Laplace-smoothed) | matches what you've been watching **lately** |
| `c_item` | ItemKNN score (cosine) | **similar to movies you watched** (evidence: top-3 contributing watched items) |
| `c_mf` | SVD/BPR-MF score | **people with tastes like yours** liked it |
| `c_pop` | `log1p(train popularity)` | **widely loved** |
| `c_trans` | `P(user moves to the item's cluster-dominant state)` | likely **next direction** of your taste (the temporal-HMM component) |

### 6.2 Learned fusion (`fusion.py`)
- Linear model `score(u,i) = Σ_k w_k · c_k(u,i)` trained with a **pairwise logistic (BPR-style) loss** on the **validation** item: for each user, positive = val item, negatives = 50–100 sampled unseen items; L2 regularization; weights ≥ 0 optional (constrained fit) so contributions are interpretable. Tune L2 by cross-validation over validation users. **Never fit on test.**
- After choosing hyperparameters, refit on train+val interactions (re-derive all features with val items added to history) and score the test item — **this is the final model** ("TasteTrace-v2"). Keep a flag to disable (`--no-refit`) for ablation.
- Optional non-linear ranker (LightGBM LambdaRank) as an *accuracy ceiling* reference only; the **linear** one is the explainable headline model.

### 6.3 Explanation generation (`explain.py`)
For item *i* shown to user *u*:
1. Contribution `contrib_k = w_k · z_k(u,i)` (z = standardized value). Pick the top-2 positive contributors.
2. Fill templates, with **concrete evidence**:
   - `c_item` → *"Because you watched **Fargo**, **Se7en** and **Heat** — all rated highly by people who liked this."* (evidence = the 3 watched items with largest `sim(i, j)`.)
   - `c_phase` → *"You're currently in a **dark-thriller phase** (state 2); this movie is typical of it (share of the phase's viewing: 18%)."* (state name from §5.1/5.2.)
   - `c_trans` → *"Your taste appears to be moving toward **drama** (probability 0.31 next)."*
   - `c_recent`, `c_mf`, `c_pop` analogous.
3. Also return a structured JSON (`components`, `contributions`, `evidence_items`, `state`, `bn_edges_used`) so the demo can plot a contribution bar chart.
Tests: explanations are deterministic; every cited evidence item is truly in the user's train history; no template placeholder left unfilled.

Acceptance: full pipeline `python scripts/run_all.py --model tastetrace_v2` produces test-set scores + explanations for every eligible user; fusion weights table saved to `results/fusion_weights.json`.

---

## 7. Phase 4 — Full evaluation, ablations, statistics (Oct 23–26)

1. **Main comparison** (P1 sampled-99 and P2 full-rank, 5 candidate seeds × 3 model-init seeds): Popularity, UserKNN, ItemKNN, SVD-MF, BPR-MF, (SASRec-lite), **KeudjeuHMM (base paper)**, HMM-only, TasteTrace-core, **TasteTrace-v2**. Table with mean ± std and bootstrap CIs; mark significance vs the base paper and vs SVD.
2. **Ablations** (drop one component at a time from the fusion; refit): −`c_phase`, −`c_trans`, −`c_recent`, −`c_item`, −`c_mf`, −`c_pop`; also *HMM replaced by static (no-phase) clustering* and *BN replaced by naive independent attributes*. Purpose: prove the **HMM and BN actually add value** (the paper's central claim) — if a component adds nothing, say so.
3. **Sensitivity**: `n_states`, `C`, history window, ESS — small line plots.
4. **Subgroup analysis**: cold-ish users (<20 interactions) vs heavy users; popular vs long-tail held-out items (does the model avoid being just a popularity bias?); report coverage and novelty.
5. **Protocol-flaw figure** (from §4.3) and **paper-protocol vs fair-protocol** table.
6. (Stretch) Repeat main comparison on **MovieLens-1M** to show generalization.

Acceptance: `results/main_comparison.csv`, `results/ablation.csv`, `results/significance.csv` exist; targets T1–T2 evaluated and stated plainly in `results/SUMMARY.md` (auto-generated: which targets met, which not).

---

## 8. Phase 4b — Explanation quality (Oct 24–27)

The explanations are the project's differentiator, so they must be **measured**.
1. **Faithfulness (deletion test).** For each user's top-10 list and each explanation citing component *k*: neutralize *k* (set `z_k := 0`) and recompute the item's rank. Metric = mean rank drop. Control = neutralize a **random other** component. Report **faithfulness rate** = fraction of explanations where removing the cited component hurts more than the control, with CI. Also a **sufficiency** check: scoring with only the cited top-2 components retains X% of the NDCG.
2. **Specificity / diversity.** Number of distinct explanation strings and the mean pairwise Jaccard similarity of explanations across different users with different histories (v1 gave identical text to everyone — must be clearly better).
3. **Coverage.** % of recommendations whose explanation includes concrete evidence items.
4. **Mini human study (strongly recommended, ~15 respondents, Google Form).** Show 10 (user history, recommendation, explanation) triples with randomized condition: TasteTrace explanation vs. generic ("popular among similar users") vs. none. Likert 1–5 on *understandability*, *trust*, *usefulness*. Save raw responses in `results/human_eval.csv`; report means and a Wilcoxon signed-rank test. (If not run in time, state it as future work — do **not** fabricate responses.)

Acceptance: `results/faithfulness.csv`, `figures/faithfulness.png`; faithfulness rate clearly above the random-control rate.

---

## 9. Required figures & tables (`scripts/make_figures.py`, all auto-generated from `results/`)
1. **Hero comparison:** grouped bars HR@10 / NDCG@10 per method with 95% CIs; base paper and TasteTrace-v2 highlighted. (Both P1 and P2 panels.)
2. **Protocol-flaw figure:** P0-vs-P1 scores for {Popularity, SVD, KeudjeuHMM, trivial-history}.
3. **Ablation bars** (Δ HR@10 per removed component).
4. **State semantics** (heatmap: HMM state × taste cluster) and **state timeline** for 3 example users.
5. **Learned BN DAGs** per state.
6. **Fusion weights** bar chart.
7. **Faithfulness** (cited-component deletion vs random control).
8. **Feature table:** capability matrix — per-user sequential modeling, collaborative signal, per-recommendation explanation, evaluation protocol (train/test split?) — rows: Base paper vs SVD vs TasteTrace-v2.

---

## 10. Phase 5 — Demo (Oct 27–29) — "see a clear comparison and demo of our model"

**Streamlit app `app/streamlit_app.py`** (`streamlit run app/streamlit_app.py`), loading cached models and precomputed results so it opens instantly. Pages:
1. **Live recommender.** Pick an existing user *or* build a profile by selecting movies in order (search box + drag order). Shows:
   - **Taste timeline:** their watch sequence colored by detected taste phase, with phase names.
   - **Current phase + next-phase probabilities** (bar chart) and the phase's BN (graph).
   - **Top-10 recommendations**, each with the generated explanation, evidence movies (posters not required — titles only), and a **component-contribution bar chart**.
   - **What-if:** remove or add a watched movie → recommendations/explanations update live.
2. **Side-by-side comparison.** Same user, three columns: **Base paper (KeudjeuHMM)** | **SVD-MF** | **TasteTrace-v2**. Highlight overlap and, for each list, whether the user's actual held-out movie appears and at what rank (for existing users). Only TasteTrace column shows explanations — make that contrast obvious.
3. **Results dashboard.** Renders the hero figure, ablations, faithfulness, protocol-flaw figure, and the main table from `results/*.csv` (no recomputation).
4. **How it works.** One diagram of the pipeline (taste clusters → HMM phases → per-phase BN → fusion → explanation).

**CLI fallback `demo.py --user 42`** prints the same story in the terminal (for presentations without a browser) and writes a markdown report to `results/demo_user_42.md`. Also export a 60–90 s screen-recording script (`docs/demo_script.md`): which user to pick, what to click, what to say.

Acceptance: app runs from a clean clone after `pip install -r requirements.txt && python scripts/download_data.py && python scripts/run_all.py`; no step requires manual edits; works offline once data is downloaded.

---

## 11. Timeline (today = Sat 10 Oct 2026; hard deadline 31 Oct)
| Dates | Phase | Exit criterion |
|---|---|---|
| Oct 10–11 | 0 Setup/refactor | smoke test reproduces v1 |
| Oct 11–12 | 1 Eval harness | baselines reproduce; oracle/random tests pass |
| Oct 12–14 | 1b Baselines + base-paper reimplementation + reproduction | `reproduction_report.md`, protocol-flaw figure |
| Oct 14–18 | 2 TasteTrace-v2 core | val gain over v1; no degenerate states |
| Oct 19–22 | 3 Fusion + explanations | final model scores + explanations |
| Oct 23–26 | 4 / 4b Evaluation, ablations, faithfulness (+ run human survey) | `SUMMARY.md` generated |
| Oct 27–29 | 5 Demo + figures | app + CLI demo working from clean clone |
| Oct 30–31 | Buffer | README polished, slides material exported, final re-run of `run_all.py` |
Buffer rule: if behind schedule, drop in this order — MovieLens-1M, SASRec-lite, LightGBM ceiling, human study (state as future work), sensitivity plots. **Never drop**: fair protocol, base-paper reimplementation, ablations, faithfulness, demo.

---

## 12. Reproducibility & quality rules
- Fixed seeds everywhere (`config.yaml: seeds`); log the config hash in every results file.
- One command regenerates everything: `python scripts/run_all.py` (stages are cached; `--force` to rerun). Total full run should be < 60 min on a laptop; print per-stage timings.
- Tune only on validation; the test item is touched once per final model. Add a test that fails if test items appear in any feature computation.
- Every claim in the README/report links to a file in `results/`. If TasteTrace-v2 does **not** beat a baseline, the summary must say so and the write-up should pivot to: *"competitive accuracy + only method with faithful per-recommendation explanations, evaluated under a leak-free protocol."*

## 13. Final deliverables checklist
- [ ] Clean repo with `README.md` (setup, one-command run, results summary), this `PLAN.md`
- [ ] `results/main_comparison.csv`, `ablation.csv`, `significance.csv`, `faithfulness.csv`, `SUMMARY.md`
- [ ] `docs/reproduction_report.md`, `docs/base_paper_assumptions.md`
- [ ] All figures in §9 under `figures/`
- [ ] Streamlit demo + CLI demo + `docs/demo_script.md`
- [ ] 1-page "claims vs evidence" table for the presentation (each claim → figure/table)

## 14. Narrative to use in the presentation (keep consistent with evidence)
1. Base paper: first to model sequential user preferences *and* movie viewer profiles with HMMs — interpretable vectors, but (a) no per-recommendation explanation, (b) evaluated without a held-out split.
2. Our reproduction: under the paper's own protocol we match their numbers **and** a trivial history-replay baseline matches them too → the protocol is not discriminative.
3. Under a fair held-out protocol, TasteTrace-v2 **outperforms the base-paper method** (CI + p-value), is competitive with matrix factorization, and ablations show the HMM phase and BN components add measurable value.
4. It is the only system here that explains each recommendation, and the explanations are **faithful** (deletion test) and **specific** (diversity metric).
5. Limitations stated openly: single dataset family (ML-100K/1M), sampled-metric caveats, explanation study sample size, no online A/B test.
