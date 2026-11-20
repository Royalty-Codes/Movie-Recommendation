# Source and reproduction boundaries

Implementation sources: the plan, section 4.2; [the authors' repository](https://github.com/HmmsMovieRecoms/HMMs-Movie-Recommendation); and [the author-posted paper](https://www.researchgate.net/publication/397477591_Hybrid_Movie_Recommendation_Based_On_Recent_User_Sequential_Preferences_and_Sequential_Profiles_Using_HMMs). The plan's repository URL omitted a hyphen. A source review resolved HMM initialization, occupation/zip thresholds and Eq.9. Repository README was inspected; source-directory/API retrieval failed through the available network, so this remains an independent Python reimplementation, not a verified execution of the authors' C program.

Implemented: separate five-state HMM per user feature and item-viewer feature, rating-bin initialization, scaled Baum–Welch, 100-step stationary marginal, concatenated vectors, Manhattan neighbors, neighbor-supported candidate set, and log importance weighted by the user's training ratings. Windows and neighbor counts are selected using validation only. P0 models are separately fitted on all ratings and cannot enter P1/P2.

Explicit assumptions:

- Initialization implements equations 1–4: raw counts divided by row total plus epsilon; each row's missing mass is distributed uniformly across its domain. Start counts divide by 1+epsilon. EM retains a numerical floor of 1e-8 to avoid singular rows.
- Rating values 1–5 map to state indices 0–4. Empty histories receive uniform feature marginals from the default initial model.
- Genre frequency counts are training rating events. Paper thresholds 700/400/100 are used when they populate all four clusters; otherwise deterministic frequency-rank quartiles populate them. Four membership bits use weights 8/4/2/1, with all-zero membership assigned the lowest word.
- Age/gender and release-year/count/mean bins follow the plan. Occupation/zip-prefix frequencies count users, with thresholds from the paper's Tables 10/12: occupation 10/20/40/70/100/150 and zip 10/50/70/100/150. All available users' public metadata is used; held-out rating values never determine these frequencies.
- The stationary vector is the average of rows of A^100 (equivalently uniform initial distribution propagated 100 times).
- Alpha in the importance equation is the user's observed training rating for that history movie. Unsupported candidates have negative infinity and receive deterministic seeded random tie breaks in evaluation.
- Eq.9 averages the user's ratings of all tied nearest history movies, multiplied by one minus nearest distance divided by the maximum catalogue item distance. The fallback adapter uses this estimate directly for unsupported candidates, while supported candidates keep Eq.11 importance. Comparing these two score scales is an explicit adapter choice. Both variants are reported; the stronger variant is chosen by validation, never by test.
- P0 measures watched-set precision/recall and binary NDCG, not rating error. HistoryReplay intentionally demonstrates that overlap evaluation rewards recommending watched movies. No RMSE/MAE headline reproduction is claimed.

The generated `reproduction_report.md` records actual deviations. Genre occurrence semantics and quantile fallback remain the plan's documented assumptions. A large P0 gap blocks a reproduction claim. Matching watched-set precision does not independently validate every implementation detail. No external code was executed and no authors' dataset was redistributed.
