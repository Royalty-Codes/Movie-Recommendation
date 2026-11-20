# 60–90 second presentation recording

1. Start `python -m streamlit run app/streamlit_app.py`. Choose viewer 42 and Live recommender. Say: “These dots show inferred viewing phases from a chronological training history.” Show the current phase, next-phase probability chart and network.
2. Open the first recommendation. Read its explanation and point to the positive contribution bars and related watched titles. Say: “The reasons are computed from the same components that ranked this movie.”
3. Remove the latest watched movie, or add a different movie. Show the updated timeline and recommendations. Explain that the model stays frozen while this profile changes.
4. Switch to Side-by-side comparison. Show the same viewer's source-reviewed HMM reimplementation, SVD and TasteTrace lists. The star marks the held-out movie only when it appears. Do not choose a viewer based on favorable results when making a population-level claim.
5. Switch to Results dashboard. Show the actual P1/P2 comparison, confidence intervals, ablations and faithfulness. State only the targets marked supported in SUMMARY.md. If a target failed, say it plainly.
6. End with limitations: single dataset, observational dependencies rather than causal effects, remaining reproduction assumptions, no human study yet.

Offline fallback: `python demo.py --user 42` writes and prints the same comparison. Smoke artifacts require `--results results/smoke` and must be labeled integration-only.
