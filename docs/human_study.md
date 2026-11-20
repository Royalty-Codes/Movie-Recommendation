# Optional human study protocol

No respondents have been recruited. No human-evaluation result is claimed.

After training, run `python scripts/human_study.py`. This exports ten fixed-seed history/recommendation trials into three counterbalanced groups. Each group sees a rotation of TasteTrace, generic and absent explanations. Present the rows in randomized trial order, hiding the condition label. Assign respondents evenly across groups and aim for at least fifteen respondents.

Ask 1–5 Likert questions about understandability, trust and usefulness. Participation should be voluntary; use anonymous respondent IDs. Collect real answers in the generated empty response template and save them as `results/human_eval.csv`. Do not fill missing values with invented answers.

Run `python scripts/human_study.py --responses results/human_eval.csv` to compute per-respondent condition means and paired two-sided Wilcoxon comparisons. Report paired respondent counts, all three dimensions, all controls and the small sample limitation. The statistical tests are exploratory and unadjusted. No automatic form publishing, respondent messages or data uploads are performed.
