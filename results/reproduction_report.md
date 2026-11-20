# Reproduction report

Computed P0 precision@10 by history window:

method
KeudjeuHMM T=100    1.0
KeudjeuHMM T=20     1.0
KeudjeuHMM T=40     1.0
KeudjeuHMM T=60     1.0
KeudjeuHMM T=80     1.0

P0 overlap precision is in the plan's expected ballpark; this alone does not verify all source equations.

Source review: https://github.com/HmmsMovieRecoms/HMMs-Movie-Recommendation and the author-posted paper linked in docs/base_paper_assumptions.md. The plan URL omitted a hyphen. Equations 1-4 and 9, plus occupation/zip thresholds, were corrected from that paper.

Full per-user evidence: results\paper_protocol.csv
No exact RMSE/MAE headline reproduction claimed.
