"""Ablation: (a) each rule group alone, (b) leave-one-group-out,
(c) word-only features (no char n-grams) raw vs. normalized.
C is tuned on dev for every condition (same protocol as the main run)."""
import pandas as pd
from load_data import load_all
from normalize import make_normalizer, ALL_GROUPS
from baseline import run_condition, identity

data = load_all()
rows = []

def add(name, fn, use_char, note):
    r, _, _ = run_condition(name, fn, use_char=use_char, save=False, data=data, verbose=False)
    r["note"] = note
    rows.append(r)
    print(f"{name:28s} dev={r['dev_macro_f1']:.4f} test_F1={r['test_macro_f1']:.4f} acc={r['test_accuracy']:.4f}", flush=True)

# (c) word-only features
add("wordonly_raw", identity, False, "word 1-2gram TF-IDF only")
add("wordonly_normalized", make_normalizer(ALL_GROUPS), False, "word 1-2gram TF-IDF only")
# (a) single groups, word+char features
add("full_raw", identity, True, "word+char")
for g in ALL_GROUPS:
    add(f"only_{g}", make_normalizer([g]), True, "single group")
# (b) leave-one-out
add("full_normalized", make_normalizer(ALL_GROUPS), True, "all groups")
for g in ALL_GROUPS:
    add(f"without_{g}", make_normalizer([x for x in ALL_GROUPS if x != g]), True, "leave-one-out")
# single groups with word-only features (where normalization should matter most)
for g in ALL_GROUPS:
    add(f"wordonly_only_{g}", make_normalizer([g]), False, "single group, word-only")

pd.DataFrame(rows).to_csv("results/ablation.csv", index=False)
print("saved results/ablation.csv")
