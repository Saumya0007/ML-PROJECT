"""Robustness / prediction-consistency evaluation.

Two sets of meaning-preserving spelling variants are generated from the TEST
sentences only (no leakage; models are NOT retrained on them):

  Set A "attested swap" : tokens that belong to a known spelling family
        (canonical form or any attested variant) are swapped for another member
        of the same family.  NOTE: the normalizer is built from the same
        families, so Set A is an optimistic / in-dictionary upper bound.
  Set B "synthetic noise": random character-level spelling noise (vowel drop,
        vowel doubling, final-letter elongation, vowel-pair / v-w / ph-f
        substitutions) that is mostly OUT of the normalizer's dictionary.
        Evaluated at several noise levels.

Consistency = share of variants whose prediction equals the prediction on the
original sentence, using the same pipeline (raw -> raw model, normalized ->
normalize(text) -> normalized model).
"""
import re, json, random
import numpy as np, pandas as pd, joblib
from sklearn.metrics import f1_score, accuracy_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from load_data import load_all, mask, SEED
from normalize import rule_table, normalize

TOK = re.compile(r"[a-z]+")
VOWELS = "aeiou"

# ---------------- Set A: attested swaps ----------------
FAMILY = {}
_fam = {}
for g, v, c in rule_table():
    _fam.setdefault(c, {c}).add(v)
for c, members in _fam.items():
    for m in members:
        FAMILY.setdefault(m, set()).update(members)


def attested_variant(text, rng, p=0.6):
    changed = [False]

    def sub(m):
        w = m.group(0)
        fam = FAMILY.get(w)
        if fam and len(fam) > 1 and rng.random() < p:
            alt = sorted(fam - {w})
            changed[0] = True
            return rng.choice(alt)
        return w
    out = TOK.sub(sub, text)
    return out if changed[0] else None


# ---------------- Set B: synthetic noise ----------------
def noisy_word(w, rng):
    ops = []
    inner_v = [i for i in range(1, len(w) - 1) if w[i] in VOWELS]
    if inner_v:
        ops += ["drop_vowel", "double_vowel"]
    ops.append("elongate")
    for a, b in (("oo", "u"), ("u", "oo"), ("ee", "i"), ("i", "ee"), ("ai", "ay"), ("v", "w"), ("w", "v"), ("ph", "f"), ("aa", "a")):
        if a in w:
            ops.append(("sub", a, b))
    op = rng.choice(ops)
    if op == "drop_vowel":
        i = rng.choice(inner_v)
        return w[:i] + w[i + 1:]
    if op == "double_vowel":
        i = rng.choice(inner_v)
        return w[:i] + w[i] + w[i:]
    if op == "elongate":
        return w + w[-1] * rng.choice([1, 2, 3])
    _, a, b = op
    return w.replace(a, b, 1)


def noisy_variant(text, rng, p):
    changed = [False]

    def sub(m):
        w = m.group(0)
        if len(w) >= 4 and w not in ("url", "user") and rng.random() < p:
            n = noisy_word(w, rng)
            if n != w:
                changed[0] = True
                return n
        return w
    # keep <url>/<user> placeholders intact
    parts = re.split(r"(<url>|<user>)", text)
    out = "".join(p_ if p_ in ("<url>", "<user>") else TOK.sub(sub, p_) for p_ in parts)
    return out if changed[0] else None


# ---------------- evaluation ----------------
def predict(ck, texts, fn):
    X = ck["features"].transform([fn(t) for t in texts])
    return ck["model"].predict(X)


def boot_ci(diff, n=2000, seed=SEED):
    rng = np.random.default_rng(seed)
    d = np.asarray(diff, dtype=float)
    m = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def evaluate(name, texts, labels, variants, ck_raw, ck_norm, subsets=None):
    """texts: original masked sentences; variants: list of (idx, variant_text)."""
    idx = np.array([i for i, _ in variants])
    vtexts = [v for _, v in variants]
    out = {"set": name, "n_variants": len(variants), "n_source_sentences": int(len(set(idx.tolist())))}
    per = {}
    for cond, ck, fn in (("raw", ck_raw, lambda t: t), ("normalized", ck_norm, normalize)):
        p_orig = predict(ck, texts, fn)
        p_var = predict(ck, vtexts, fn)
        same = (p_var == p_orig[idx]).astype(int)
        y = np.asarray(labels)[idx]
        per[cond] = same
        out[cond] = {
            "consistency": round(float(same.mean()), 4),
            "macro_f1_variants": round(f1_score(y, p_var, average="macro"), 4),
            "accuracy_variants": round(accuracy_score(y, p_var), 4),
            "macro_f1_originals_same_rows": round(f1_score(y, p_orig[idx], average="macro"), 4),
        }
    if subsets:
        out["subsets"] = {k: {"n": int(m.sum()), "raw_consistency": round(float(per["raw"][m].mean()), 4),
                              "norm_consistency": round(float(per["normalized"][m].mean()), 4)}
                          for k, m in subsets.items() if m.sum() > 0}
    diff = per["normalized"] - per["raw"]
    lo, hi = boot_ci(diff)
    out["consistency_gain_norm_minus_raw"] = round(float(diff.mean()), 4)
    out["gain_95ci"] = [round(lo, 4), round(hi, 4)]
    return out


if __name__ == "__main__":
    _, _, test = load_all()
    texts = [mask(t) for t in test.text]
    labels = test.label.values
    ck_raw = joblib.load("checkpoints/raw.joblib")
    ck_norm = joblib.load("checkpoints/normalized.joblib")
    results = []

    # Set A
    rng = random.Random(SEED)
    A = []
    for i, t in enumerate(texts):
        for _ in range(3):
            v = attested_variant(t, rng)
            if v is not None and v != t:
                A.append((i, v))
    NEG_FAM = FAMILY["nahi"]
    def touches_negation(i, v):
        return any((a != b) and (a in NEG_FAM or b in NEG_FAM) for a, b in zip(TOK.findall(texts[i]), TOK.findall(v)))
    neg_mask = np.array([touches_negation(i, v) for i, v in A])
    results.append(evaluate("A_attested_swap", texts, labels, A, ck_raw, ck_norm,
                            subsets={"touches_negation_family": neg_mask, "other_families": ~neg_mask}))

    # Set B at several noise levels
    samples = []
    for p in (0.1, 0.2, 0.3, 0.5):
        rng = random.Random(SEED + int(p * 100))
        B = []
        for i, t in enumerate(texts):
            for _ in range(3):
                v = noisy_variant(t, rng, p)
                if v is not None and v != t:
                    B.append((i, v))
        r = evaluate(f"B_synthetic_noise_p{p}", texts, labels, B, ck_raw, ck_norm)
        r["noise_p"] = p
        results.append(r)
        if p == 0.3:
            samples = B

    # sample files for manual meaning-preservation check
    rs = random.Random(SEED)
    pick = rs.sample(A, 60)
    pd.DataFrame([{"set": "A", "original": texts[i], "variant": v} for i, v in pick] +
                 [{"set": "B_p0.3", "original": texts[i], "variant": v} for i, v in rs.sample(samples, 60)]
                 ).to_csv("results/robustness_variant_sample.csv", index=False)

    json.dump(results, open("results/robustness.json", "w"), indent=2)
    for r in results:
        print(r["set"], "n=", r["n_variants"], "| raw cons", r["raw"]["consistency"],
              "| norm cons", r["normalized"]["consistency"], "| gain", r["consistency_gain_norm_minus_raw"], r["gain_95ci"])

    # plot
    B = [r for r in results if r["set"].startswith("B_")]
    ps = [r["noise_p"] for r in B]
    plt.figure(figsize=(4.2, 3))
    plt.plot(ps, [r["raw"]["consistency"] for r in B], "o-", label="raw")
    plt.plot(ps, [r["normalized"]["consistency"] for r in B], "s-", label="normalized")
    plt.xlabel("per-word noise probability"); plt.ylabel("prediction consistency")
    plt.ylim(0.5, 1.0); plt.grid(alpha=.3); plt.legend(); plt.tight_layout()
    plt.savefig("results/robustness_curve.png", dpi=200)
