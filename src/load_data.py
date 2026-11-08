import pandas as pd


def load_conll(path):
    """Read a SentiMix CoNLL-style file and rebuild one sentence per example."""
    rows, toks, uid, lab = [], [], None, None
    for line in open(path, encoding="utf-8"):
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


def load_all(d="data"):
    """Load train, dev, and test (with test labels merged in by uid)."""
    train = load_conll(f"{d}/Hinglish_train_14k_split_conll.txt")
    dev = load_conll(f"{d}/Hinglish_dev_3k_split_conll.txt")
    test = load_conll(f"{d}/Hinglish_test_unalbelled_conll_updated.txt").drop(columns="label")
    labels = pd.read_csv(f"{d}/Hinglish_test_labels.txt")
    labels["Uid"] = labels["Uid"].astype(str)
    test = (
        test.merge(labels, left_on="uid", right_on="Uid")
        .rename(columns={"Sentiment": "label"})
        .drop(columns="Uid")
    )
    return train, dev, test


if __name__ == "__main__":
    tr, dv, te = load_all()
    print(len(tr), len(dv), len(te))
    print(pd.concat([tr, dv, te]).label.value_counts())