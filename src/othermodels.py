"""Does the raw-vs-normalized conclusion hold for other classifiers?
Same data, same features, same protocol as baseline.py (hyper-parameter chosen on DEV macro-F1, test once)."""
import json, pandas as pd
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import ComplementNB
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score, accuracy_score
from load_data import load_all, mask, make_features, SEED
from normalize import make_normalizer, ALL_GROUPS

MODELS = {
    "LinearSVC":    (lambda p: LinearSVC(C=p, random_state=SEED, max_iter=5000), [0.05, 0.1, 0.25, 0.5, 1]),
    "ComplementNB": (lambda p: ComplementNB(alpha=p),                           [0.1, 0.3, 1.0]),
    "SGD-modhuber": (lambda p: SGDClassifier(loss="modified_huber", alpha=p, random_state=SEED, max_iter=50, tol=None), [1e-5, 3e-5, 1e-4]),
}
train, dev, test = load_all()
norm = make_normalizer(ALL_GROUPS)
rows = []
for cond, fn in (("raw", lambda t: t), ("normalized", norm)):
    prep = lambda s: [fn(mask(t)) for t in s]
    feats = make_features(True)
    Ftr = feats.fit_transform(prep(train.text)); Fdv = feats.transform(prep(dev.text)); Fte = feats.transform(prep(test.text))
    for name, (mk, grid) in MODELS.items():
        best = max(grid, key=lambda p: f1_score(dev.label, mk(p).fit(Ftr, train.label).predict(Fdv), average="macro"))
        m = mk(best).fit(Ftr, train.label); pred = m.predict(Fte)
        rows.append({"model": name, "condition": cond, "best_param": best,
                     "test_macro_f1": round(f1_score(test.label, pred, average="macro"), 4),
                     "test_accuracy": round(accuracy_score(test.label, pred), 4)})
        print(rows[-1], flush=True)
df = pd.DataFrame(rows); df.to_csv("results/other_models.csv", index=False)
print(df.pivot(index="model", columns="condition", values="test_macro_f1").assign(diff=lambda d: d.normalized - d.raw).round(4))