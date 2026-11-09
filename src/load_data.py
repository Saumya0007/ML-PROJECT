"""Shared utilities: data loading, masking, feature construction."""
import re
import pandas as pd
from sklearn.pipeline import make_union
from sklearn.feature_extraction.text import TfidfVectorizer

SEED = 42
DATA_DIR = "data"


def load_conll(path):
    """Read a SentiMix CoNLL-style file; rebuild one sentence per example."""
    rows, toks, uid, lab = [], [], None, None
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("meta\t"):
                if uid is not None and toks:
                    rows.append((uid, " ".join(toks), lab))
                p = line.split("\t")
                uid, lab, toks = p[1], (p[2] if len(p) > 2 else None), []
            elif line.strip():
                toks.append(line.split("\t")[0])
    if uid is not None and toks:
        rows.append((uid, " ".join(toks), lab))
    return pd.DataFrame(rows, columns=["uid", "text", "label"])


def load_all(d=DATA_DIR):
    train = load_conll(f"{d}/Hinglish_train_14k_split_conll.txt")
    dev = load_conll(f"{d}/Hinglish_dev_3k_split_conll.txt")
    test = load_conll(f"{d}/Hinglish_test_unalbelled_conll_updated.txt").drop(columns="label")
    labels = pd.read_csv(f"{d}/Hinglish_test_labels.txt")
    labels["Uid"] = labels["Uid"].astype(str)
    test = (test.merge(labels, left_on="uid", right_on="Uid")
                .rename(columns={"Sentiment": "label"}).drop(columns="Uid"))
    return train, dev, test


def mask(text):
    """Cleaning applied identically to the raw and normalized conditions."""
    text = text.lower()
    text = re.sub(r"https?\s*:?\s*//\s*(?:\S+\s*[/.]\s*)*\S+", " <url> ", text)
    text = re.sub(r"@\s*\w+", " <user> ", text)
    return re.sub(r"\s+", " ", text).strip()


def make_features(use_char=True):
    parts = [TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)]
    if use_char:
        parts.append(TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=3,
                                     max_features=150000, sublinear_tf=True))
    return make_union(*parts)


if __name__ == "__main__":
    tr, dv, te = load_all()
    print(len(tr), len(dv), len(te))
    print(pd.concat([tr, dv, te]).label.value_counts())
