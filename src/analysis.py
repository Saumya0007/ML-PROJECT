"""Class-wise analysis, subset analysis and error analysis (test set)."""
import re, json
import numpy as np, pandas as pd
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, f1_score, accuracy_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from load_data import mask, SEED
from normalize import GROUPS, NEGATION_TOKENS, normalize

TOK = re.compile(r"[a-z]+")
LABELS = ["negative", "neutral", "positive"]
raw = pd.read_csv("results/raw_predictions.csv")
nrm = pd.read_csv("results/normalized_predictions.csv")
assert (raw.uid == nrm.uid).all()
y = raw.true.values

# ---- class-wise ----
rows = []
for cond, p in (("raw", raw.pred.values), ("normalized", nrm.pred.values)):
    P, R, F, S = precision_recall_fscore_support(y, p, labels=LABELS, zero_division=0)
    for i, l in enumerate(LABELS):
        rows.append({"condition": cond, "class": l, "precision": P[i], "recall": R[i], "f1": F[i], "support": int(S[i])})
cw = pd.DataFrame(rows)
cw.round(4).to_csv("results/classwise.csv", index=False)
piv = cw.pivot(index="class", columns="condition", values="f1")
piv["delta_f1"] = piv["normalized"] - piv["raw"]
piv.round(4).to_csv("results/classwise_delta.csv")
for cond, p in (("raw", raw.pred.values), ("normalized", nrm.pred.values)):
    pd.DataFrame(confusion_matrix(y, p, labels=LABELS), index=LABELS, columns=LABELS).to_csv(f"results/confusion_{cond}.csv")

fig, ax = plt.subplots(figsize=(4.2, 2.8))
w = 0.35
x = np.arange(3)
ax.bar(x - w / 2, piv.loc[LABELS, "raw"], w, label="raw")
ax.bar(x + w / 2, piv.loc[LABELS, "normalized"], w, label="normalized")
ax.set_xticks(x); ax.set_xticklabels(LABELS); ax.set_ylabel("F1"); ax.set_ylim(0.5, 0.8); ax.legend()
plt.tight_layout(); plt.savefig("results/classwise_f1.png", dpi=200)

# ---- flags ----
variant_spell = set(v for g in ("negation", "spelling") for vs in GROUPS[g].values() for v in vs)
variant_slang = set(v for g in ("abbreviation", "english_slang") for vs in GROUPS[g].values() for v in vs)
NEG = NEGATION_TOKENS | {"nhi", "ni", "nai"}

def flags(t):
    toks = TOK.findall(t)
    return {
        "spelling_variant": any(w in variant_spell for w in toks),
        "slang_abbrev": any(w in variant_slang for w in toks) or bool(re.search(r"([a-z])\1{2,}", t)),
        "negation": any(w in NEG for w in toks),
    }

masked = [mask(t) for t in raw.text]
F = pd.DataFrame([flags(t) for t in masked])
changed = np.array([normalize(t) != t for t in masked])
F["changed_by_norm"] = changed
rc, nc = raw.pred.values == y, nrm.pred.values == y

# ---- subset analysis: where does normalization matter? ----
sub = {}
for name, m in (("changed_by_normalization", changed), ("unchanged", ~changed),
                ("has_negation", F.negation.values), ("has_spelling_variant", F.spelling_variant.values),
                ("has_slang_abbrev", F.slang_abbrev.values)):
    sub[name] = {"n": int(m.sum()),
                 "raw_acc": round(float(rc[m].mean()), 4), "norm_acc": round(float(nc[m].mean()), 4),
                 "raw_macro_f1": round(f1_score(y[m], raw.pred.values[m], average="macro"), 4),
                 "norm_macro_f1": round(f1_score(y[m], nrm.pred.values[m], average="macro"), 4)}

# ---- error analysis (normalized-pipeline and raw errors) ----
errs_raw = np.where(~rc)[0]
def primary_raw(i):
    if F.spelling_variant[i]:
        return "spelling variation"
    if F.slang_abbrev[i]:
        return "slang/abbreviation"
    if F.negation[i]:
        return "negation"
    return "other (sarcasm, context, neutral ambiguity)"
cats_raw = pd.Series([primary_raw(i) for i in errs_raw]).value_counts().to_dict()

summary = {
    "n_test": int(len(y)),
    "errors_raw": int((~rc).sum()), "errors_normalized": int((~nc).sum()),
    "primary_category_of_raw_errors": cats_raw,
    "error_rate_with_flag_raw": {k: round(float((~rc)[F[k].values].mean()), 4) for k in ("spelling_variant", "slang_abbrev", "negation")},
    "error_rate_without_flag_raw": {k: round(float((~rc)[~F[k].values].mean()), 4) for k in ("spelling_variant", "slang_abbrev", "negation")},
    "fixed_by_normalization": int((~rc & nc).sum()),
    "broken_by_normalization": int((rc & ~nc).sum()),
    "broken_with_negation_in_sentence": int((rc & ~nc & F.negation.values).sum()),
    # decomposition: rewritten sentences vs sentences the rules did not touch
    "changed_sentences": {"n": int(changed.sum()), "fixed": int((~rc & nc & changed).sum()),
                          "broken": int((rc & ~nc & changed).sum()),
                          "prediction_flips": int(((raw.pred.values != nrm.pred.values) & changed).sum()),
                          "broken_with_negation": int((rc & ~nc & changed & F.negation.values).sum()),
                          "fixed_with_negation": int((~rc & nc & changed & F.negation.values).sum()),
                          "flips_with_negation": int(((raw.pred.values != nrm.pred.values) & changed & F.negation.values).sum()),
                          "with_negation": int((changed & F.negation.values).sum())},
    "unchanged_sentences": {"n": int((~changed).sum()), "fixed": int((~rc & nc & ~changed).sum()),
                            "broken": int((rc & ~nc & ~changed).sum()),
                            "prediction_flips": int(((raw.pred.values != nrm.pred.values) & ~changed).sum())},
    "subsets": sub,
}
# retraining-churn reference: raw pipeline refit with another C (no text change at all)
import joblib
from sklearn.linear_model import LogisticRegression
from load_data import load_all
_tr, _dv, _te = load_all()
ck = joblib.load("checkpoints/raw.joblib")
Ftr = ck["features"].transform([mask(t) for t in _tr.text]); Fte = ck["features"].transform(masked)
alt = LogisticRegression(C=0.5, max_iter=1000, random_state=SEED).fit(Ftr, _tr.label).predict(Fte)
summary["churn_reference_raw_C025_vs_C05_flip_pct"] = round(100 * float((alt != raw.pred.values).mean()), 2)
summary["flip_pct_changed_sentences"] = round(100 * summary["changed_sentences"]["prediction_flips"] / summary["changed_sentences"]["n"], 2)
summary["flip_pct_unchanged_sentences"] = round(100 * summary["unchanged_sentences"]["prediction_flips"] / summary["unchanged_sentences"]["n"], 2)
# neutral-class share of errors
summary["share_of_raw_errors_with_true_neutral"] = round(float((raw.true.values[~rc] == "neutral").mean()), 4)
summary["share_of_raw_errors_with_pred_neutral"] = round(float((raw.pred.values[~rc] == "neutral").mean()), 4)
json.dump(summary, open("results/error_summary.json", "w"), indent=2)

# sample of interesting cases
out = pd.DataFrame({"uid": raw.uid, "text": masked, "normalized": [normalize(t) for t in masked], "true": y,
                    "raw_pred": raw.pred, "norm_pred": nrm.pred, **{k: F[k] for k in F.columns}})
broken = out[(rc & ~nc & changed)]; fixed = out[(~rc & nc & changed)]
churn = out[((rc & ~nc) | (~rc & nc)) & ~changed].head(40)
other_err = out[(~rc & ~nc)].sample(40, random_state=SEED)
pd.concat([broken.assign(case="broken_by_norm"), fixed.assign(case="fixed_by_norm"),
           other_err.assign(case="error_both"), churn.assign(case="retraining_churn_unchanged_text")]).to_csv("results/error_analysis_sample.csv", index=False)
print(json.dumps(summary, indent=2))
print(piv.round(4))
