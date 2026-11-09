"""Main experiment: raw vs. spelling-normalized text, same pipeline.

Protocol: C is selected on the DEV set only (macro-F1); the model is then
refit with that C and evaluated ONCE on the held-out TEST set.
"""
import os, json, time, argparse
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report

from load_data import load_all, mask, make_features, SEED
from normalize import make_normalizer, ALL_GROUPS

C_GRID = [0.1, 0.25, 0.5, 1, 2]


def identity(t):
    return t


def run_condition(name, normalize_fn, use_char=True, C=None, save=True, data=None, verbose=True):
    """Train/evaluate one condition. If C is None, tune on dev."""
    train, dev, test = data if data is not None else load_all()
    prep = lambda s: [normalize_fn(mask(t)) for t in s]
    Xtr, Xdv, Xte = prep(train.text), prep(dev.text), prep(test.text)

    feats = make_features(use_char)
    Ftr = feats.fit_transform(Xtr)                 # vectorizer fit on TRAIN only
    Fdv, Fte = feats.transform(Xdv), feats.transform(Xte)

    dev_scores = {}
    grid = [C] if C is not None else C_GRID
    for c in grid:
        m = LogisticRegression(C=c, max_iter=1000, random_state=SEED).fit(Ftr, train.label)
        dev_scores[c] = f1_score(dev.label, m.predict(Fdv), average="macro")
        if verbose:
            print(f"  [{name}] C={c:<5} dev macro-F1={dev_scores[c]:.4f}", flush=True)
    best_C = max(dev_scores, key=dev_scores.get)

    model = LogisticRegression(C=best_C, max_iter=1000, random_state=SEED).fit(Ftr, train.label)
    pred = model.predict(Fte)
    res = {
        "condition": name, "use_char_ngrams": use_char, "best_C": best_C,
        "dev_macro_f1": round(dev_scores[best_C], 4),
        "test_macro_f1": round(f1_score(test.label, pred, average="macro"), 4),
        "test_accuracy": round(accuracy_score(test.label, pred), 4),
        "n_features": int(Ftr.shape[1]),
        "dev_macro_f1_by_C": {str(k): round(v, 4) for k, v in dev_scores.items()},
    }
    if verbose:
        print(json.dumps(res))
    if save:
        os.makedirs("results", exist_ok=True)
        os.makedirs("checkpoints", exist_ok=True)
        json.dump(res, open(f"results/{name}.json", "w"), indent=2)
        pd.DataFrame({"uid": test.uid, "text": test.text, "true": test.label, "pred": pred}) \
          .to_csv(f"results/{name}_predictions.csv", index=False)
        with open(f"results/{name}_classification_report.txt", "w") as f:
            f.write(classification_report(test.label, pred, digits=4))
        joblib.dump({"features": feats, "model": model}, f"checkpoints/{name}.joblib", compress=3)
    return res, feats, model


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["raw", "normalized"], default=None)
    a = ap.parse_args()
    data = load_all()
    t0 = time.time()
    if a.only in (None, "raw"):
        run_condition("raw", identity, data=data)
    if a.only in (None, "normalized"):
        run_condition("normalized", make_normalizer(ALL_GROUPS), data=data)
    print(f"done in {time.time()-t0:.0f}s")
