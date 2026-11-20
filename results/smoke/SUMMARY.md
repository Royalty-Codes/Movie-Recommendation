# TasteTrace results

Config hash: `e84d278fd6adbe92`

**SMOKE RUN: integration evidence only; not the final experiment.**

T1 — beats validation-selected base-paper variant (KeudjeuHMM), both P1 metrics with p < .01: **False**.

T2 — difference CI versus SVD includes zero or favors TasteTrace on both P1 metrics: **True**.

T3 — faithfulness rate lower 95% bound > .5: **True**.

T4 — cached Streamlit and CLI demo, comparison figures: generated; see README for commands.

Core acceptance diagnostics: {
  "config_hash": "e84d278fd6adbe92",
  "bn_all_user_catalog_seconds": 0.04578789998777211,
  "mean_top10_duplicate_fraction": 0.7850000000000001,
  "ties_under_5_percent": false
}


## Held-out comparison (computed)

             method protocol  metric     mean  std   ci_low  ci_high  users      config_hash
             BPR-MF       P1   HR@10 0.420000  0.0 0.330000 0.520000    100 e84d278fd6adbe92
             BPR-MF       P1 NDCG@10 0.236888  0.0 0.181196 0.304474    100 e84d278fd6adbe92
             BPR-MF       P2   HR@10 0.070000  0.0 0.030000 0.130000    100 e84d278fd6adbe92
             BPR-MF       P2 NDCG@10 0.043106  0.0 0.016181 0.078051    100 e84d278fd6adbe92
           HMM-only       P1   HR@10 0.180000  0.0 0.110000 0.250000    100 e84d278fd6adbe92
           HMM-only       P1 NDCG@10 0.096713  0.0 0.053649 0.147105    100 e84d278fd6adbe92
           HMM-only       P2   HR@10 0.050000  0.0 0.010000 0.100000    100 e84d278fd6adbe92
           HMM-only       P2 NDCG@10 0.023520  0.0 0.005000 0.047191    100 e84d278fd6adbe92
            ItemKNN       P1   HR@10 0.540000  0.0 0.450000 0.640000    100 e84d278fd6adbe92
            ItemKNN       P1 NDCG@10 0.302464  0.0 0.242708 0.374830    100 e84d278fd6adbe92
            ItemKNN       P2   HR@10 0.100000  0.0 0.050000 0.170000    100 e84d278fd6adbe92
            ItemKNN       P2 NDCG@10 0.043380  0.0 0.019308 0.074525    100 e84d278fd6adbe92
         KeudjeuHMM       P1   HR@10 0.570000  0.0 0.480000 0.665250    100 e84d278fd6adbe92
         KeudjeuHMM       P1 NDCG@10 0.310217  0.0 0.250194 0.369302    100 e84d278fd6adbe92
         KeudjeuHMM       P2   HR@10 0.110000  0.0 0.050000 0.170000    100 e84d278fd6adbe92
         KeudjeuHMM       P2 NDCG@10 0.055882  0.0 0.025157 0.091362    100 e84d278fd6adbe92
KeudjeuHMM+fallback       P1   HR@10 0.570000  0.0 0.480000 0.665250    100 e84d278fd6adbe92
KeudjeuHMM+fallback       P1 NDCG@10 0.310217  0.0 0.250194 0.369302    100 e84d278fd6adbe92
KeudjeuHMM+fallback       P2   HR@10 0.110000  0.0 0.050000 0.170000    100 e84d278fd6adbe92
KeudjeuHMM+fallback       P2 NDCG@10 0.055882  0.0 0.025157 0.091362    100 e84d278fd6adbe92
         Popularity       P1   HR@10 0.480000  0.0 0.390000 0.580000    100 e84d278fd6adbe92
         Popularity       P1 NDCG@10 0.252381  0.0 0.194725 0.313988    100 e84d278fd6adbe92
         Popularity       P2   HR@10 0.050000  0.0 0.010000 0.090000    100 e84d278fd6adbe92
         Popularity       P2 NDCG@10 0.023655  0.0 0.003708 0.047639    100 e84d278fd6adbe92
             SVD-MF       P1   HR@10 0.520000  0.0 0.430000 0.620000    100 e84d278fd6adbe92
             SVD-MF       P1 NDCG@10 0.293979  0.0 0.233600 0.365015    100 e84d278fd6adbe92
             SVD-MF       P2   HR@10 0.090000  0.0 0.040000 0.150000    100 e84d278fd6adbe92
             SVD-MF       P2 NDCG@10 0.051195  0.0 0.019947 0.092511    100 e84d278fd6adbe92
    TasteTrace-core       P1   HR@10 0.420000  0.0 0.330000 0.520000    100 e84d278fd6adbe92
    TasteTrace-core       P1 NDCG@10 0.203039  0.0 0.152220 0.266299    100 e84d278fd6adbe92
    TasteTrace-core       P2   HR@10 0.040000  0.0 0.010000 0.090000    100 e84d278fd6adbe92
    TasteTrace-core       P2 NDCG@10 0.028869  0.0 0.005000 0.069463    100 e84d278fd6adbe92
      TasteTrace-v2       P1   HR@10 0.600000  0.0 0.510000 0.695250    100 e84d278fd6adbe92
      TasteTrace-v2       P1 NDCG@10 0.330224  0.0 0.270790 0.397685    100 e84d278fd6adbe92
      TasteTrace-v2       P2   HR@10 0.110000  0.0 0.054750 0.170000    100 e84d278fd6adbe92
      TasteTrace-v2       P2 NDCG@10 0.048937  0.0 0.022807 0.080326    100 e84d278fd6adbe92
            UserKNN       P1   HR@10 0.580000  0.0 0.490000 0.670000    100 e84d278fd6adbe92
            UserKNN       P1 NDCG@10 0.300090  0.0 0.243231 0.365796    100 e84d278fd6adbe92
            UserKNN       P2   HR@10 0.080000  0.0 0.030000 0.140000    100 e84d278fd6adbe92
            UserKNN       P2 NDCG@10 0.043367  0.0 0.015733 0.079929    100 e84d278fd6adbe92


## Interpretation

No superiority claim is supported unless its paired difference and permutation test pass. A zero or harmful ablation delta means that component has not demonstrated added accuracy. This is one dataset, with one held-out positive per user; Recall@10 therefore equals HR@10. P0 overlap precision is not comparable to P1 hit rate. The source-paper fallback equation could not be verified; see assumptions.


## Missing external evidence

Human study not conducted; no responses fabricated. MovieLens-1M, SASRec and non-linear ranker are stretch work, not included in this run.


## Evidence

main_comparison.csv; significance.csv; ablation.csv; faithfulness.csv; paper_protocol.csv; run_manifest.json.