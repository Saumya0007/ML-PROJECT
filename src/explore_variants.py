import re
from collections import Counter, defaultdict
from load_data import load_all
from baseline import mask

train, _, _ = load_all("data")
cnt = Counter()
for t in train.text:
    cnt.update(re.findall(r"[a-z]+", mask(t)))

def key(w):
    w = re.sub(r"(.)\1+", r"\1", w)      # collapse repeated letters
    return re.sub(r"[aeiouyh]", "", w)   # drop vowels and h

groups = defaultdict(list)
for w, c in cnt.items():
    k = key(w)
    if c >= 5 and k:
        groups[k].append((w, c))

out = [sorted(v, key=lambda x: -x[1]) for v in groups.values() if len(v) > 1]
out.sort(key=lambda v: -sum(c for _, c in v))
with open("results/variant_candidates.txt", "w", encoding="utf-8") as f:
    for v in out[:300]:
        f.write(" | ".join(f"{w}({c})" for w, c in v) + "\n")