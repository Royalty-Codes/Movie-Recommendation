# TasteTrace

Temporal movie recommendations with collaborative taste clusters, a shared categorical HMM, per-phase Bayesian networks and a validation-trained explainable linear ranker.

## Run

Python 3.10–3.12, CPU. From this folder:

```powershell
python -m pip install -r requirements.txt
python scripts/download_data.py
python -m pytest -q
python -u scripts/run_all.py
python -m streamlit run app/streamlit_app.py
```

CLI presentation fallback: `python demo.py --user 42`.

The run caches intermediate models by configuration, source and input-data hashes. Re-running resumes compatible stages; `--force` rebuilds them. `--no-refit` disables final train+validation refitting. `--model tastetrace_v2` is accepted for the plan's command; shared dependencies and comparison models still run. Hyperparameter grids and seeds are in `config.yaml`. The run uses four CPU threads; ML-100K does not need a GPU/Kaggle account. Runtime is measured in `results/timings.csv`; do not assume a promised runtime on another machine.

Quick integration run (100 users, reduced grids, separate outputs):

```powershell
python scripts/run_all.py --smoke
python demo.py --user 42 --results results/smoke
$env:TASTETRACE_RESULTS = 'results/smoke'
python -m streamlit run app/streamlit_app.py
```

Remove that environment variable to use full results: `Remove-Item Env:TASTETRACE_RESULTS`.

## Evidence

Read [the generated summary](results/SUMMARY.md) for actual outcomes. Main evidence is [comparison](results/main_comparison.csv), [paired statistics](results/significance.csv), [refitted ablations](results/ablation.csv), [ablation statistics](results/ablation_significance.csv), [faithfulness](results/faithfulness.csv), [P0 overlap evaluation](results/paper_protocol.csv) and [run configuration](results/run_manifest.json). Numerical claims must come from these files; smoke results are integration checks only.

`scripts/make_figures.py` regenerates comparison, protocol-flaw, ablation, state semantics/timeline, DAG, weights, faithfulness, sensitivity and capability figures. The app provides a viewing timeline, next-phase probabilities, learned DAG, explained top-10 lists, ordered custom profiles with drag controls, what-if add/remove, side-by-side comparisons and a results dashboard. Models remain frozen during profile edits.

## Evaluation and limitations

Per-user timestamp order retains original file order for ties. Last interaction is test, second-last is validation; users with fewer than five interactions stay in training. Item counts, means, embeddings, clusters, HMMs and BNs use the current training partition only. Hyperparameters use validation. Fusion uses grouped-user cross-validation of pairwise loss on validation positives and unseen negatives. After selection, **all methods** refit on train+validation for the test comparison, giving them identical history access. Test values never train the models.

P1 uses one held-out positive and 99 unseen negatives with shared candidates/tie keys across methods. P2 ranks against all unseen items. Full-run methods use five candidate seeds and three model initialization seeds; P2 has one candidate set because its catalogue is fixed. Seed metrics are averaged within user before bootstrap CIs and paired tests. Recall@10 equals HR@10 with one positive. Coverage and mean training popularity describe the top-10 lists. Main CIs are over users; initialization standard deviations are separate. P-values are two-sided, unadjusted planned comparisons, not familywise guarantees.

The base-paper reimplementation was reviewed against the author-posted paper. The plan's GitHub URL omitted a hyphen; [the correct repository](https://github.com/HmmsMovieRecoms/HMMs-Movie-Recommendation) was found during source review. HMM initialization, occupation/zip grouping and Eq.9 follow that paper; remaining genre-frequency and held-out adapter assumptions are explicit in [source-method notes](docs/base_paper_assumptions.md). This is an independent Python implementation, not a verified execution of the authors' C code. [The reproduction report](docs/reproduction_report.md) records actual P0 gaps. P0 overlap precision and P1 held-out HR measure different tasks. The historical v1 benchmark cannot be verified within ±.01 without its missing benchmark/candidate settings; preserved v1 code is evaluated under the new protocol instead.

“Temporal-causal” in the original project title is a motivation, **not an identified causal result**. BN edges encode observational dependencies. Faithfulness tests component deletion; it does not establish human usefulness. No human responses are fabricated. ML-1M loaders/downloader are included, but ML-1M experiments, SASRec and non-linear ceilings are optional stretch work and are not claimed completed.

## Project files

- `src/tastetrace/`: data, splits, features, clustering, HMM, BN, fusion, explanations, baselines and evaluation.
- `scripts/run_all.py`: resumable experiment, statistics, reports and figures.
- `app/streamlit_app.py`, `demo.py`: interactive and terminal demos.
- `tests/`: chronology, leakage, shared candidates, oracle/random scoring, CPD lookup, fusion and evidence tests.
- [Original plan](PLAN.md), [claims versus evidence](docs/claims_vs_evidence.md), [demo script](docs/demo_script.md).

The original `main.py` and four `src/*.py` files are retained for v1 compatibility. No MovieLens data should be committed or redistributed. The downloader fetches official GroupLens archives; read the dataset's license before use. Once artifacts and data exist, the demo works offline.

The optional human study has an [export and analysis protocol](docs/human_study.md). Run `python scripts/human_study.py` to export counterbalanced stimuli and an empty response template. Actual human participation remains future work until real responses are collected.
