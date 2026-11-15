# Robust Hinglish Sentiment Analysis: Does Spelling Normalization Help?

Controlled evaluation of dictionary-based spelling normalization for Roman-script Hinglish sentiment
(SentiMix / SemEval-2020 Task 9, Hinglish track), using an identical TF-IDF + logistic regression
pipeline on raw vs. normalized text.
Authors: Saumyadip Deb, Ankit Kumar, Anand Kumar Sah (IIT Jodhpur).

## Headline results (test set, 3,000 tweets)
| Condition | Dev Macro-F1 | Test Macro-F1 | Test Acc |
|---|---|---|---|
| Raw | 0.6249 | 0.6865 | 0.6813 |
| Normalized | 0.6306 | 0.6877 | 0.6827 |

- No significant difference (McNemar p = 0.82; Macro-F1 diff +0.0012, 95% CI [-0.0073, +0.0096]).
- Consistency on in-dictionary variants 92.1% -> 100% (by construction); no gain on out-of-dictionary noise.
- Rules touch only 2.4% of tokens; a small gain (+0.008 F1) appears only with 10% of the training data.
- Audit: 0 negation tokens lost, 0 rule violations; manual sample of 100 sentences: 94 preserved, 4 not preserved, 2 unsure.

Full details: `report/main.pdf`.

## Repository structure
```
README.md  requirements.txt  requirements-lock.txt  run_all.py  .gitignore
data/          README.md only (dataset NOT committed; download instructions inside)
src/           load_data.py normalize.py baseline.py ablation.py learning_curve.py robustness.py
               audit.py analysis.py vocab_stats.py significance.py explore_variants.py
               make_report_tables.py commit_log.py
results/       metrics (*.json/*.csv), predictions, audit samples, plots, commit_log.txt
checkpoints/   raw.joblib, normalized.joblib (~2 MB each, committed)
report/        main.tex, main.pdf, gen/ (auto-generated tables)
```

## Setup
```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements-lock.txt   # exact versions; requirements.txt = minimum versions
```

## Data
See `data/README.md`. Then verify:
```bash
python src/load_data.py     # must print: 14000 3000 3000, neutral 7492, positive 6616, negative 5892
```

## Reproduce everything (CPU only, about 15 min)
Run all commands from the project root.
```bash
python run_all.py
```
Or step by step: `python src/baseline.py` (about 40 s), then `ablation.py`, `learning_curve.py`,
`significance.py`, `robustness.py`, `audit.py`, `analysis.py`, `vocab_stats.py`, `make_report_tables.py`.

Build the report (needs LaTeX, or upload `report/` to Overleaf):
```bash
cd report && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## Commit hashes and timestamps
```bash
python src/commit_log.py        # writes results/commit_log.txt and report/gen/tab_commits.tex
git log --pretty=format:"%h | %ad | %s" --date=iso
```
The commit table in the report (Table 15) is generated from this.

## Checkpoints
`checkpoints/*.joblib` hold the fitted TF-IDF vectorizers and logistic-regression models
(~2 MB each, small enough to commit). They were produced with scikit-learn 1.9.1; with another
version re-create them with `python src/baseline.py`.

## Reproducibility declaration
All reported numbers are produced by the scripts in this repository on the official SentiMix splits.
The pipeline is CPU-only and deterministic: fixed seed 42 (classifier, subsampling, variant generation,
bootstrap), `PYTHONHASHSEED=0`, L-BFGS solver, no GPU. The regularisation strength C is chosen on the
development set only and the test set is evaluated once per condition. Library versions are pinned in
`requirements-lock.txt`. Re-running `src/baseline.py` reproduces the reported test predictions exactly;
tiny differences (about 0.0003 in a dev score at C=2) may occur across library versions.
The dataset is public and not redistributed here.