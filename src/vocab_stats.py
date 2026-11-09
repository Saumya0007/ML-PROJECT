"""Vocabulary / OOV statistics: how much does normalization shrink the vocabulary?"""
import re, json
from collections import Counter
from load_data import load_all, mask
from normalize import normalize, build_lookup, ALL_GROUPS

TOKEN = re.compile(r"(?u)\b\w\w+\b")      # same pattern as TfidfVectorizer
train, dev, test = load_all()
look = build_lookup(ALL_GROUPS)

def stats(fn):
    tr = [TOKEN.findall(fn(mask(t))) for t in train.text]
    te = [TOKEN.findall(fn(mask(t))) for t in test.text]
    df = Counter(w for doc in tr for w in set(doc))
    v1, v2 = set(df), {w for w, c in df.items() if c >= 2}
    n_te = sum(len(d) for d in te)
    return {
        "train_types": len(v1), "train_types_df2": len(v2),
        "test_oov_rate_df1_pct": round(100 * sum(w not in v1 for d in te for w in d) / n_te, 3),
        "test_oov_rate_df2_pct": round(100 * sum(w not in v2 for d in te for w in d) / n_te, 3),
        "train_tokens": sum(len(d) for d in tr),
    }

raw, nrm = stats(lambda t: t), stats(normalize)
toks = [w for t in train.text for w in TOKEN.findall(mask(t))]
res = {"raw": raw, "normalized": nrm,
       "variant_token_share_pct": round(100 * sum(w in look for w in toks) / len(toks), 2),
       "types_reduction_pct": round(100 * (1 - nrm["train_types"] / raw["train_types"]), 2),
       "types_df2_reduction_pct": round(100 * (1 - nrm["train_types_df2"] / raw["train_types_df2"]), 2)}
json.dump(res, open("results/vocab_stats.json", "w"), indent=2)
print(json.dumps(res, indent=2))
