"""Find candidate spelling-variant groups in the TRAIN split (used to build the rules).
Groups words that share the same consonant skeleton; the output must be reviewed by hand
(unrelated words sharing a skeleton are grouped too)."""
import re, os
from collections import Counter, defaultdict
from load_data import load_all, mask

train, _, _ = load_all()
cnt = Counter()
for t in train.text:
    cnt.update(re.findall(r"[a-z]+", mask(t)))

def key(w):
    w = re.sub(r"(.)\1+", r"\1", w)       # collapse repeated letters
    return re.sub(r"[aeiouyh]", "", w)    # drop vowels and h

groups = defaultdict(list)
for w, c in cnt.items():
    k = key(w)
    if c >= 5 and k:
        groups[k].append((w, c))
out = sorted((sorted(v, key=lambda x: -x[1]) for v in groups.values() if len(v) > 1),
             key=lambda v: -sum(c for _, c in v))
os.makedirs("results", exist_ok=True)
with open("results/variant_candidates.txt", "w", encoding="utf-8") as f:
    for v in out[:300]:
        f.write(" | ".join(f"{w}({c})" for w, c in v) + "\n")
print("wrote results/variant_candidates.txt")
