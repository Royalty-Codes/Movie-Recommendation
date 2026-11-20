# Claims versus evidence

| Proposed claim | Evidence required | Decision rule |
|---|---|---|
| Beats the source-reviewed base-paper implementation | results/significance.csv; figures/hero_comparison.png | Positive paired differences and p < .01 on both P1 metrics, against validation-selected variant |
| Competitive with SVD | results/main_comparison.csv; results/significance.csv | Report paired CI; overlapping marginal CIs alone do not prove equivalence |
| HMM/BN add accuracy | results/ablation.csv; figures/ablations.png | Full score exceeds each relevant refitted ablation; otherwise say not demonstrated |
| Explanations affect ranking | results/faithfulness.csv; figures/faithfulness.png | Cited deletion hurts more than control, user-bootstrap CI reported |
| Evidence is grounded in viewing history | tests; results/explanations.jsonl | Every evidence ID belongs to that user's training/refit history |
| P0 rewards history replay | results/paper_protocol.csv; figures/protocol_flaw.png | Report observed overlap precision, distinguish it from held-out HR |
| Source-paper headline reproduced | results/reproduction_report.md; docs/base_paper_assumptions.md | Source equations reviewed; report measured P0 agreement and remaining assumptions, never claim exact C-code parity |
| Human usefulness or causal effects | No evidence collected | Do not claim; future work |

All numerical claims must cite artifacts from the same run's config hash. Smoke outputs are not final research evidence.
