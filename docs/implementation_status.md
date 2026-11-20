# Implementation status

The ML-100K research pipeline, prescribed classical baselines, source-reviewed paper reimplementation and fallback adapter, P0/P1/P2 evaluation, paired statistics, refitted ablations, sensitivity/subgroup outputs, explanations, figures and both demos are implemented. The original v1 code remains available through a package adapter. All experiments are script-generated; no evaluation numbers are hand-entered.

See `results/SUMMARY.md` for actual target and acceptance outcomes, and `results/verification.json` for the artifact audit. A failed empirical acceptance check stays failed; implementation completion is not a claim that every research hypothesis succeeded.

Explicit limits:

- The discrete BN score still ties on shared attribute combinations. The plan's <5% tie expectation is checked and reported. No random perturbation is added to manufacture unique BN scores.
- Historical v1 ±.01 reproduction is not independently established without the missing original benchmark configuration and candidate pools. A fair new-split v1 comparison is produced instead.
- Source equations were reviewed from the author-posted paper, and the corrected repository README was consulted. Executing the authors' C code was not verified; remaining genre-count/adapter assumptions are documented.
- Embedding dimension is selected using validation MF first, followed by joint cluster-count/state-count/ESS selection. This staged search avoids the full Cartesian product. Recent-window sensitivity is descriptive on validation cross-validation loss. Fusion coefficients selected before final refitting remain frozen across final model initialization seeds.
- Final refits include validation history for every model, so comparison histories are aligned. The no-refit ablation is a first-seed diagnostic.
- Human-study materials and analysis are implemented; real participation is outstanding. No survey was sent and no responses were fabricated.
- MovieLens-1M download/load support exists. ML-1M experiments, SASRec-lite and the non-linear accuracy ceiling are the plan's optional stretch items and were omitted to prioritize the required pipeline.

Local CPU training was sufficient. Kaggle access was not needed. Temporary app testing uses a headless localhost server; the normal launch command is in README.
