import re

# canonical form -> spelling variants (lowercase). Review every entry by hand.
RULES = {
    "nahi":  ["nahin", "nhi", "nahii", "nahee"],
    "accha": ["acha", "achha", "achcha", "acchaa"],
    "bahut": ["bohot", "bohut", "bahot", "bhut"],
    "kya":   ["kyaa", "kia"],
    "hai":   ["hai", "hain"],
}

# negation words: never merge an ambiguous word into these, and never delete them
NEGATION = {"nahi", "nahin", "nhi", "mat", "na", "nai"}

LOOKUP = {v: canon for canon, vs in RULES.items() for v in vs}

def normalize(text):
    return re.sub(r"[a-z]+", lambda m: LOOKUP.get(m.group(0), m.group(0)), text)