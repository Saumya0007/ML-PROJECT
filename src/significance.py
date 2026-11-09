"""Paired significance tests: raw vs. normalized predictions on the test set."""
import json
import numpy as np, pandas as pd
from scipy.stats import chi2
from sklearn.metrics import f1_score
from load_data import SEED

a = pd.read_csv("results/raw_predictions.csv")
b = pd.read_csv("results/normalized_predictions.csv")
assert (a.uid == b.uid).all()
y, pa, pb = a.true.values, a.pred.values, b.pred.values

ca, cb = pa == y, pb == y
only_norm = int(((~ca) & cb).sum())     # normalized right, raw wrong
only_raw = int((ca & (~cb)).sum())      # raw right, normalized wrong
stat = (abs(only_norm - only_raw) - 1) ** 2 / max(only_norm + only_raw, 1)
p_mcnemar = float(1 - chi2.cdf(stat, 1))

rng = np.random.default_rng(SEED)
d_f1, d_acc = [], []
for _ in range(2000):
    i = rng.integers(0, len(y), len(y))
    d_f1.append(f1_score(y[i], pb[i], average="macro") - f1_score(y[i], pa[i], average="macro"))
    d_acc.append(cb[i].mean() - ca[i].mean())
res = {
    "n_test": int(len(y)),
    "raw_only_correct": only_raw, "normalized_only_correct": only_norm,
    "both_correct": int((ca & cb).sum()), "both_wrong": int(((~ca) & (~cb)).sum()),
    "predictions_changed": int((pa != pb).sum()),
    "mcnemar_p": round(p_mcnemar, 4),
    "macro_f1_diff_mean": round(float(np.mean(d_f1)), 4),
    "macro_f1_diff_95ci": [round(float(np.percentile(d_f1, 2.5)), 4), round(float(np.percentile(d_f1, 97.5)), 4)],
    "accuracy_diff_mean": round(float(np.mean(d_acc)), 4),
    "accuracy_diff_95ci": [round(float(np.percentile(d_acc, 2.5)), 4), round(float(np.percentile(d_acc, 97.5)), 4)],
}
json.dump(res, open("results/significance.json", "w"), indent=2)
print(json.dumps(res, indent=2))
