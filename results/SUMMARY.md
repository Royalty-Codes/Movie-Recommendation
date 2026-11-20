# TasteTrace experiment results

Run hash: `6a2677c20837731d`. Dataset: ml-100k.

5 candidate seeds × 3 model initialization seeds; 943 evaluated users. P2 uses the complete candidate catalogue once per initialization.

## Held-out accuracy

Mean with 95% user-bootstrap interval. All methods receive the same chronological history and candidate sets.

| Method | P1 HR@10 | P1 NDCG@10 | P2 HR@10 | P2 NDCG@10 |
|---|---:|---:|---:|---:|
| TasteTrace-v2 | 0.701 [0.675, 0.727] | 0.415 [0.395, 0.436] | 0.154 [0.133, 0.175] | 0.081 [0.068, 0.094] |
| KeudjeuHMM | 0.569 [0.539, 0.598] | 0.329 [0.309, 0.350] | 0.109 [0.090, 0.129] | 0.054 [0.043, 0.065] |
| SVD-MF | 0.630 [0.603, 0.658] | 0.377 [0.356, 0.398] | 0.143 [0.122, 0.164] | 0.073 [0.061, 0.086] |
| BPR-MF | 0.643 [0.617, 0.669] | 0.374 [0.355, 0.395] | 0.134 [0.115, 0.153] | 0.067 [0.057, 0.078] |
| ItemKNN | 0.606 [0.578, 0.635] | 0.346 [0.326, 0.367] | 0.118 [0.098, 0.139] | 0.064 [0.052, 0.077] |
| UserKNN | 0.593 [0.564, 0.623] | 0.346 [0.325, 0.367] | 0.121 [0.100, 0.142] | 0.061 [0.049, 0.072] |
| Popularity | 0.411 [0.382, 0.441] | 0.228 [0.208, 0.248] | 0.086 [0.069, 0.105] | 0.044 [0.034, 0.055] |
| TasteTrace-core | 0.514 [0.487, 0.540] | 0.262 [0.245, 0.279] | 0.058 [0.047, 0.071] | 0.027 [0.021, 0.033] |
| HMM-only | 0.458 [0.433, 0.482] | 0.229 [0.215, 0.244] | 0.047 [0.037, 0.058] | 0.021 [0.016, 0.027] |
| HistoryReplay | 0.106 [0.097, 0.115] | 0.047 [0.043, 0.051] | 0.006 [0.002, 0.012] | 0.002 [0.001, 0.004] |

## Targets

- T1: MET — both P1 metrics exceed the validation-selected paper variant (KeudjeuHMM) with paired permutation p < .01.
- T2: MET — the paired difference versus SVD on P1 is favorable or includes zero on both metrics; this is not an equivalence test.
- T3: MET — cited-component deletion win-rate lower 95% bound exceeds .5.
- T4: cached Streamlit and CLI demos, structured explanations and comparison figures are generated; see README for launch commands.

## Paired comparisons

P-values are two-sided and unadjusted. Differences average sampling and initialization seeds within each user before inference.

| Comparator | Protocol / metric | Difference | Paired 95% CI | p |
|---|---|---:|---|---:|
| KeudjeuHMM | P1 / HR@10 | +0.1324 | [+0.1078, +0.1572] | 0.00010 |
| SVD-MF | P1 / HR@10 | +0.0712 | [+0.0543, +0.0885] | 0.00010 |
| BPR-MF | P1 / HR@10 | +0.0583 | [+0.0407, +0.0759] | 0.00010 |
| KeudjeuHMM | P1 / NDCG@10 | +0.0855 | [+0.0702, +0.1009] | 0.00010 |
| SVD-MF | P1 / NDCG@10 | +0.0383 | [+0.0290, +0.0477] | 0.00010 |
| BPR-MF | P1 / NDCG@10 | +0.0405 | [+0.0292, +0.0518] | 0.00010 |
| KeudjeuHMM | P2 / HR@10 | +0.0449 | [+0.0247, +0.0647] | 0.00010 |
| SVD-MF | P2 / HR@10 | +0.0113 | [-0.0007, +0.0230] | 0.07709 |
| BPR-MF | P2 / HR@10 | +0.0205 | [+0.0053, +0.0361] | 0.01200 |
| KeudjeuHMM | P2 / NDCG@10 | +0.0272 | [+0.0149, +0.0394] | 0.00010 |
| SVD-MF | P2 / NDCG@10 | +0.0076 | [+0.0011, +0.0141] | 0.02200 |
| BPR-MF | P2 / NDCG@10 | +0.0137 | [+0.0050, +0.0226] | 0.00300 |

## Component evidence

The static-phase ablation removes temporal conditioning; the independent-BN ablation tests dependencies. Each ranker is refitted on validation. No-refit is a first-initialization diagnostic.

| Ablation | Metric | Full minus ablation | 95% CI | p |
|---|---|---:|---|---:|
| independent_bn | HR@10 | +0.0046 | [-0.0003, +0.0093] | 0.06129 |
| independent_bn | NDCG@10 | +0.0040 | [+0.0016, +0.0065] | 0.00080 |
| no_refit | HR@10 | +0.0177 | [+0.0050, +0.0310] | 0.00950 |
| no_refit | NDCG@10 | +0.0130 | [+0.0058, +0.0204] | 0.00070 |
| static | HR@10 | +0.0251 | [+0.0146, +0.0358] | 0.00010 |
| static | NDCG@10 | +0.0171 | [+0.0117, +0.0225] | 0.00010 |
| without_c_item | HR@10 | +0.0028 | [-0.0031, +0.0088] | 0.37136 |
| without_c_item | NDCG@10 | -0.0030 | [-0.0062, +0.0001] | 0.06259 |
| without_c_mf | HR@10 | +0.0260 | [+0.0152, +0.0367] | 0.00010 |
| without_c_mf | NDCG@10 | +0.0269 | [+0.0196, +0.0343] | 0.00010 |
| without_c_phase | HR@10 | +0.0090 | [+0.0023, +0.0157] | 0.00930 |
| without_c_phase | NDCG@10 | +0.0044 | [+0.0011, +0.0077] | 0.01030 |
| without_c_pop | HR@10 | +0.0052 | [+0.0004, +0.0100] | 0.04090 |
| without_c_pop | NDCG@10 | +0.0046 | [+0.0022, +0.0070] | 0.00040 |
| without_c_recent | HR@10 | +0.0150 | [+0.0069, +0.0231] | 0.00050 |
| without_c_recent | NDCG@10 | +0.0099 | [+0.0055, +0.0143] | 0.00010 |
| without_c_trans | HR@10 | +0.0065 | [+0.0010, +0.0120] | 0.02180 |
| without_c_trans | NDCG@10 | +0.0061 | [+0.0032, +0.0090] | 0.00010 |

Added accuracy is **not demonstrated** for: independent_bn (HR@10), without_c_item (HR@10), without_c_item (NDCG@10).

## Explanations

Cited-component deletion mean rank drop: 36.66; random-other control: 5.50.

Deletion win rate: 98.160%, 95% CI [97.937%, 98.367%]. Both positive cited components are assessed when present.

Concrete watched-title evidence coverage: 99.989%. Distinct texts: 4897 / 9430. Cross-user mean text Jaccard: 0.402.

Sufficiency NDCG ratio using only the target explanation's cited components: 1.346. Ratios above one mean the reduced score happened to rank targets better.

Deletion favors components selected for large positive contributions. This checks arithmetic consistency, not causal validity or human usefulness.

## Acceptance checks and limitations

Selected validation core: 24 clusters, 8 requested states (8 effective), ESS=5; validation HR@10=0.575.

Catalogue BN scoring over all users: 0.206 seconds. Mean duplicate fraction in core top-10 scores: 78.6%. The <5% core-tie target is NOT MET. Fusion retains continuous collaborative signals; discrete BN attributes still tie.

Historical v1 ±.01 reproduction is not verified because exact original candidate pools and benchmark code were not supplied. The new split is different. The preserved v1 core is evaluated on this split; see v1_validation_comparison.json and legacy_v1.csv.

The base implementation follows reviewed paper equations, with genre-frequency and held-out adapter choices documented in docs/base_paper_assumptions.md. Matching P0 overlap precision alone does not verify exact authors-code parity. Read reproduction_report.md.

One dataset family and one held-out positive per user; Recall@10 equals HR@10. P0 watched-set precision and P1 held-out HR measure different tasks. Bayesian edges express statistical dependencies, not identified causal effects.

Human study has not been conducted. Optional materials are available through scripts/human_study.py; no responses are fabricated. MovieLens-1M experiments, SASRec and a non-linear ranker remain optional stretch work.

## Files

main_comparison.csv · significance.csv · ablation.csv · ablation_significance.csv · faithfulness.csv · protocol_comparison.csv · run_manifest.json. Figures are in the configured figures directory.

