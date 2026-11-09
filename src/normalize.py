"""Conservative spelling normalization for Roman-script Hinglish.

Rules are organised in groups so each group can be switched on/off for the
ablation study.  Design principles
  * Token-level dictionary lookup only (no sentence rewriting).
  * A variant is mapped only if it almost always means the SAME word.
  * Ambiguous forms are deliberately EXCLUDED (see AMBIGUOUS_EXCLUDED), e.g.
    'na', 'ni', 'nai' (negation vs. other meanings), 'ache' (English 'ache'
    vs. Hindi 'achhe'), 'kia' (kiya vs. kya), 'bs', 'hm', 'tm', 'sale'.
  * Negation words are only ever mapped onto other negation words.
"""
import re

# canonical form -> list of variants (all lowercase)
NEGATION = {
    "nahi": ["nahin", "nhi", "nehi", "nahii", "nahee", "nhin", "nahiin", "nehin", "nhii", "nahe"],
}

SPELLING = {
    "achha": ["acha", "accha", "achcha", "aacha", "achaa", "acchaa", "aachha"],
    "achhi": ["achi", "acchi", "aachi", "achhii"],
    "achhe": ["acche", "aache", "achhey"],
    "bahut": ["bohot", "bohut", "bahot", "bohat", "boht", "bhot", "bhaut", "bhout", "bahoot", "bahat"],
    "kuch": ["kuchh"],
    "kyun": ["kyu", "kyon", "kyo", "kiu", "kyoon"],
    "kyunki": ["kyuki", "kyunke", "kyonki", "kyuke"],
    "hain": ["hein"],
    "hoon": ["hun"],
    "theek": ["thik", "thek", "theak"],
    "jaise": ["jese", "jaisay"],
    "jaisa": ["jesa"],
    "jaisi": ["jesi"],
    "kaise": ["kese", "kaisay"],
    "kaisa": ["kesa"],
    "kaisi": ["kesi"],
    "pehle": ["pahle", "pehlay", "phele"],
    "sath": ["saath"],
    "tarah": ["tarha", "tarhan"],
    "aapko": ["apko", "appko"],
    "aapki": ["apki", "appki"],
    "aapke": ["apke", "appke"],
    "aapka": ["apka", "appka"],
    "pyaar": ["pyar"],
    "yaar": ["yar"],
    "pura": ["poora"],
    "jhoot": ["jhut", "jhuth", "juth", "jhooth"],
    "chahiye": ["chaiye", "chahye", "chaahiye", "chahie", "chahiyee"],
    "chutiya": ["chutia", "chootiya", "chutiyaa"],
    "ghatiya": ["ghatia", "ghatiyaa"],
    "hamesha": ["humesha", "hamesa", "humesa", "hmesha", "hmesa"],
    "umeed": ["umid", "ummid", "ummeed"],
    "shaadi": ["shadi"],
    "shayad": ["sayad", "shayed"],
    "zindagi": ["jindagi", "zindgi", "zindagee", "jindgi"],
    "zyada": ["jyada", "jyda", "zyda", "jada", "zada"],
    "mujhe": ["mujhey", "mujhay"],
    "waala": ["wala"],   # canonical 'wala' handled below via inversion check
    "bhi": ["bhee", "bhii"],
    "abhi": ["abhii"],
    "kabhi": ["kabi"],
    "woh": ["wo", "vo", "voh", "wooh"],
    "toh": ["tou"],
    "hamare": ["humare", "hamaare"],
    "hamara": ["humara", "hamaara"],
    "hamari": ["humari", "hamaari"],
    "isliye": ["isiliye", "islie"],
    "khush": ["khus", "khushh"],
    "saala": ["sala"],
    "behen": ["behan", "bahen", "bahan"],
    "dhokha": ["dhoka"],
    "pani": ["paani"],
    "dekho": ["dekhoo"],
}
# 'waala' above would invert the common 'wala'; fix by making 'wala' canonical.
SPELLING.pop("waala")
SPELLING["wala"] = ["waala"]
SPELLING["wale"] = ["waale"]
SPELLING["wali"] = ["waali"]

ABBREVIATION = {   # vowel-dropped / shortened Hinglish forms
    "bahut": ["bht"],
    "kar": ["kr"],
    "karna": ["krna", "kerna"],
    "karne": ["krne"],
    "karte": ["krte", "kerte"],
    "karta": ["krta", "kerta"],
    "karti": ["krti"],
    "karo": ["kro"],
    "kare": ["kre"],
    "karke": ["krke"],
    "hai": ["h"],
    "haan": ["hn"],
    "pata": ["pta"],
    "phir": ["phr"],
    "agar": ["agr"],
    "ghar": ["ghr"],
    "sirf": ["srf"],
    "kabhi": ["kbhi", "kbi"],
    "tumhe": ["tmhe"],
    "mujhe": ["mjhe"],
    "sab": ["sb"],
    "sakta": ["skta"],
    "sakte": ["skte", "skty"],
    "gaya": ["gya"],
    "gayi": ["gyi"],
    "gaye": ["gye"],
    "tak": ["tk"],
    "jab": ["jb"],
    "kab": ["kb"],
    "kuch": ["kch"],
    "raha": ["rha"],
    "rahe": ["rhe"],
    "rahi": ["rhi"],
    "laga": ["lga"],
    "bata": ["bta"],
    "bataya": ["btaya"],
    "tarah": ["trah", "trha", "tarh"],
}

ENGLISH_SLANG = {
    "good": ["gud"],
    "love": ["luv"],
    "please": ["plz", "pls", "plss", "plzz"],
    "your": ["ur"],
    "you": ["u"],
    "thanks": ["thanx", "thnx", "tnx", "thx", "thanku"],
    "birthday": ["bday"],
    "because": ["bcz", "coz", "bcoz", "cuz"],
    "tomorrow": ["tmrw", "tomorow"],
    "what": ["wat", "wht"],
    "with": ["wid"],
    "about": ["abt"],
    "people": ["ppl", "pple"],
    "sorry": ["sry"],
    "friend": ["frnd"],
    "friends": ["frnds"],
    "time": ["tym"],
    "awesome": ["awsm"],
    "nice": ["nyc"],
    "happy": ["hapy"],
}

GROUPS = {
    "negation": NEGATION,
    "spelling": SPELLING,
    "abbreviation": ABBREVIATION,
    "english_slang": ENGLISH_SLANG,
}
ALL_GROUPS = ["negation", "spelling", "abbreviation", "english_slang", "elongation"]

# Words deliberately NOT normalized (documented in the report)
AMBIGUOUS_EXCLUDED = ["na", "ni", "nai", "naa", "nah", "mat", "mt", "ache", "kia", "bs", "hm", "tm",
                      "m", "s", "t", "k", "pr", "hen", "sale", "kam", "km", "tho", "bi", "abi", "ap", "ky", "shi"]

# All tokens that express negation (original + canonical) - used only by the audit
NEGATION_TOKENS = {"nahi", "nahin", "nhi", "nehi", "nahii", "nahee", "nhin", "nahiin", "nehin", "nhii",
                   "nahe", "na", "ni", "nai", "naa", "nah", "mat", "mt", "no", "not", "never", "nope",
                   "nahin", "neither", "nor", "cannot", "dont", "cant", "wont"}


def build_lookup(groups):
    look = {}
    for g in groups:
        if g not in GROUPS:
            continue
        for canon, variants in GROUPS[g].items():
            for v in variants:
                look[v] = canon
    return look


_TOKEN = re.compile(r"[a-z]+")
_ELONG = re.compile(r"([a-z])\1{2,}")


def make_normalizer(groups=ALL_GROUPS):
    look = build_lookup(groups)
    elong = "elongation" in groups

    def normalize(text):
        def sub(m):
            w = m.group(0)
            if elong:
                w = _ELONG.sub(r"\1\1", w)       # 'nahiiii' -> 'nahii'
            return look.get(w, w)
        return _TOKEN.sub(sub, text)
    return normalize


normalize = make_normalizer()      # default: all groups


def rule_table():
    """Flat list of (group, variant, canonical)."""
    return [(g, v, c) for g, d in GROUPS.items() for c, vs in d.items() for v in vs]


def _self_check():
    seen = {}
    for g, v, c in rule_table():
        assert " " not in v, f"multi-word variant not supported: {v}"
        assert v != c, (g, v, c)
        if v in seen and seen[v] != c:
            raise AssertionError(f"conflicting rules for {v}: {seen[v]} vs {c}")
        seen[v] = c
    for v, c in seen.items():
        assert c not in seen, f"chained rule {v}->{c}->{seen[c]}"
    # negation words may only map to negation words, never the reverse
    for g, v, c in rule_table():
        assert (v in NEGATION_TOKENS) == (c in NEGATION_TOKENS), f"negation mismatch {v}->{c}"
    for w in AMBIGUOUS_EXCLUDED:
        assert w not in seen, f"excluded word has a rule: {w}"
    return len(seen)


if __name__ == "__main__":
    print("rules OK:", _self_check(), "variant forms")
