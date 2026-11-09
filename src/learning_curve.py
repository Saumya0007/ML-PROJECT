"""Data-efficiency: does normalization help more when training data is scarce?
Fixed C=0.5 (value selected for the full-data run); random train subsets, 3 seeds
(1 for the full set).  Evaluated on the test set; dev only used for nothing here."""
import numpy as np, pandas as pd
from load_data import load_all
from normalize import make_normalizer, ALL_GROUPS
from baseline import run_condition, identity

train, dev, test = load_all()
norm = make_normalizer(ALL_GROUPS)
rows = []
for frac in (0.1, 0.25, 0.5, 1.0):
    seeds = (42, 43, 44) if frac < 1 else (42,)
    for s in seeds:
        sub = train if frac == 1.0 else train.sample(frac=frac, random_state=s)
        out = {}
        for name, fn in (("raw", identity), ("normalized", norm)):
            r, _, _ = run_condition(name, fn, C=0.5, save=False, data=(sub, dev, test), verbose=False)
            out[name] = r["test_macro_f1"]
        rows.append({"fraction": frac, "n_train": len(sub), "seed": s, "raw": out["raw"], "normalized": out["normalized"]})
        print(rows[-1], flush=True)
df = pd.DataFrame(rows)
df["diff"] = df.normalized - df.raw
df.to_csv("results/learning_curve.csv", index=False)
print(df.groupby("fraction")[["raw", "normalized", "diff"]].agg(["mean", "std"]).round(4))
