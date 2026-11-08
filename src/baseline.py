import re
import json
import os
import pandas as pd
from sklearn.pipeline import make_pipeline, make_union
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report

from load_data import load_all


def mask(text):
    """Basic cleaning applied to BOTH raw and normalized conditions."""
    text = text.lower()
    # URLs are split into tokens in this dataset, e.g. "https // t . co / AbC123"
    text = re.sub(r"https?\s*:?\s*//\s*(?:\S+\s*[/.]\s*)*\S+", " <url> ", text)
    # usernames appear as "@ username"
    text = re.sub(r"@\s*\w+", " <user> ", text)
    return re.sub(r"\s+", " ", text).strip()


def identity(text):
    return text  # raw condition: no spelling normalization


def build_model(C):
    features = make_union(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True),
    )
    return make_pipeline(features, LogisticRegression(C=C, max_iter=2000))


def run(normalize_fn, name):
    train, dev, test = load_all("data")
    prep = lambda s: s.map(lambda t: normalize_fn(mask(t)))
    Xtr, Xdv, Xte = prep(train.text), prep(dev.text), prep(test.text)

    # 1) choose C on the DEV set only
    best_C, best_f1 = None, -1
    for C in [0.5, 1, 2, 5, 10]:
        m = build_model(C).fit(Xtr, train.label)
        f1 = f1_score(dev.label, m.predict(Xdv), average="macro")
        print(f"C={C:<4} dev macro-F1={f1:.4f}")
        if f1 > best_f1:
            best_C, best_f1 = C, f1

    # 2) with C fixed, evaluate on TEST once
    model = build_model(best_C).fit(Xtr, train.label)
    pred = model.predict(Xte)
    res = {
        "condition": name,
        "best_C": best_C,
        "dev_macro_f1": round(best_f1, 4),
        "test_macro_f1": round(f1_score(test.label, pred, average="macro"), 4),
        "test_accuracy": round(accuracy_score(test.label, pred), 4),
    }
    print(json.dumps(res, indent=2))
    print(classification_report(test.label, pred, digits=3))

    os.makedirs("results", exist_ok=True)
    json.dump(res, open(f"results/{name}.json", "w"), indent=2)
    pd.DataFrame({"uid": test.uid, "text": test.text, "true": test.label, "pred": pred}) \
      .to_csv(f"results/{name}_predictions.csv", index=False)
    return model


if __name__ == "__main__":
    from normalize import normalize
    run(identity, "raw")
    run(normalize, "normalized")