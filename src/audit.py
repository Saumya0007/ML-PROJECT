"""Sentiment-preservation audit of the normalizer.

(1) Rule audit      : every rule checked for negation compatibility + corpus frequency.
(2) Sentence audit  : on all splits, count sentences changed by normalization and
                      check that no negation token is lost or created.
(3) Manual sample   : 100 random changed TEST sentences written to
                      results/audit_sample.csv with an empty `auditor_label` column
                      (preserved / not_preserved / unsure).  Fill it in, save as
                      results/audit_sample_labeled.csv and re-run this script.
"""
import re, json, os
from collections import Counter
import numpy as np, pandas as pd
from load_data import load_all, mask, SEED
from normalize import rule_table, normalize, NEGATION_TOKENS, make_normalizer, GROUPS

TOK = re.compile(r"[a-z]+")
train, dev, test = load_all()
allx = pd.concat([train.assign(split="train"), dev.assign(split="dev"), test.assign(split="test")])
allx["masked"] = allx.text.map(mask)
allx["norm"] = allx.masked.map(normalize)

# ---- (1) rule audit ----
freq = Counter()
for t in allx.masked:
    freq.update(TOK.findall(t))
rows = []
for g, v, c in rule_table():
    ctx = ""
    for t in allx.masked:
        m = re.search(rf"\b{re.escape(v)}\b", t)
        if m:
            ctx = t[max(0, m.start() - 30): m.end() + 30]
            break
    rows.append({"group": g, "variant": v, "canonical": c, "freq_all_splits": freq[v],
                 "variant_is_negation": v in NEGATION_TOKENS, "canonical_is_negation": c in NEGATION_TOKENS,
                 "negation_compatible": (v in NEGATION_TOKENS) == (c in NEGATION_TOKENS), "example_context": ctx})
rule_df = pd.DataFrame(rows).sort_values("freq_all_splits", ascending=False)
rule_df.to_csv("results/rule_audit.csv", index=False)

# ---- (2) sentence audit ----
def neg_count(t):
    return sum(w in NEGATION_TOKENS for w in TOK.findall(t))

allx["changed"] = allx.masked != allx.norm
allx["neg_before"] = allx.masked.map(neg_count)
allx["neg_after"] = allx.norm.map(neg_count)
ch = allx[allx.changed]
res = {
    "rules_total": len(rule_df),
    "rules_observed_in_data": int((rule_df.freq_all_splits > 0).sum()),
    "rules_negation_incompatible": int((~rule_df.negation_compatible).sum()),
    "sentences_total": int(len(allx)),
    "sentences_changed": int(allx.changed.sum()),
    "sentences_changed_pct": round(100 * allx.changed.mean(), 2),
    "test_sentences_changed": int(allx[(allx.split == "test")].changed.sum()),
    "negation_tokens_lost": int((ch.neg_after < ch.neg_before).sum()),
    "negation_tokens_recovered_by_normalization": int((ch.neg_after > ch.neg_before).sum()),
    "negation_count_unchanged": int((ch.neg_after == ch.neg_before).sum()),
    "mean_tokens_changed_per_changed_sentence": round(float(np.mean([
        sum(a != b for a, b in zip(TOK.findall(x), TOK.findall(y))) for x, y in zip(ch.masked, ch.norm)])), 2),
}
for g in GROUPS:
    n = make_normalizer([g])
    res[f"sentences_changed_by_{g}"] = int((allx.masked.map(n) != allx.masked).sum())
res["sentences_changed_by_elongation"] = int((allx.masked.map(make_normalizer(["elongation"])) != allx.masked).sum())
# top rules by impact
res["top10_rules"] = [f"{r.variant}->{r.canonical} ({r.freq_all_splits})" for r in rule_df.head(10).itertuples()]

# ---- (3) manual-audit sample ----
tc = allx[(allx.split == "test") & allx.changed]
samp = tc.sample(min(100, len(tc)), random_state=SEED)
def diff(x, y):
    return "; ".join(f"{a}->{b}" for a, b in zip(TOK.findall(x), TOK.findall(y)) if a != b)
out = pd.DataFrame({"uid": samp.uid, "gold_label": samp.label, "original": samp.masked, "normalized": samp.norm,
                    "changed_tokens": [diff(x, y) for x, y in zip(samp.masked, samp.norm)],
                    "has_negation_token": samp.neg_before > 0, "auditor_label": "", "notes": ""})
out.to_csv("results/audit_sample.csv", index=False)
if os.path.exists("results/audit_sample_labeled.csv"):
    lab = pd.read_csv("results/audit_sample_labeled.csv")
    vc = lab.auditor_label.value_counts().to_dict()
    res["manual_audit_counts"] = vc
    res["manual_audit_n"] = int(sum(vc.values()))
json.dump(res, open("results/audit.json", "w"), indent=2)
print(json.dumps(res, indent=2))
