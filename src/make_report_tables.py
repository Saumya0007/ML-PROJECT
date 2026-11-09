"""Generate LaTeX tables + macros for the report directly from results/.
Every number in the report is produced here, so the PDF always matches the code."""
import json, os
import pandas as pd
from load_data import load_all

R = "results"
OUT = "report/gen"
os.makedirs(OUT, exist_ok=True)
J = lambda f: json.load(open(f"{R}/{f}"))
f3 = lambda x: f"{x:.3f}"
pct = lambda x, d=1: f"{100*x:.{d}f}"

raw, nrm = J("raw.json"), J("normalized.json")
sig, aud, err, rob = J("significance.json"), J("audit.json"), J("error_summary.json"), J("robustness.json")
abl = pd.read_csv(f"{R}/ablation.csv").set_index("condition")
cw = pd.read_csv(f"{R}/classwise.csv")
train, dev, test = load_all()

# ---------------- macros ----------------
m = {}
def mac(name, val): m[name] = str(val)
mac("RawDevF", f3(raw["dev_macro_f1"])); mac("NormDevF", f3(nrm["dev_macro_f1"]))
f4 = lambda x: f"{x:.4f}"
mac("RawTestF", f4(raw["test_macro_f1"])); mac("NormTestF", f4(nrm["test_macro_f1"]))
mac("RawTestAcc", f3(raw["test_accuracy"])); mac("NormTestAcc", f3(nrm["test_accuracy"]))
mac("RawC", raw["best_C"]); mac("NormC", nrm["best_C"])
mac("McNemarP", f"{sig['mcnemar_p']:.2f}")
mac("FOnlyNorm", sig["normalized_only_correct"]); mac("FOnlyRaw", sig["raw_only_correct"])
mac("PredChanged", sig["predictions_changed"])
lo, hi = sig["macro_f1_diff_95ci"]
mac("FDiffCI", f"[{lo:+.4f}, {hi:+.4f}]"); mac("FDiff", f"{nrm['test_macro_f1']-raw['test_macro_f1']:+.4f}")
mac("RulesTotal", aud["rules_total"]); mac("RulesObserved", aud["rules_observed_in_data"])
mac("ChangedPct", f"{aud['sentences_changed_pct']:.1f}"); mac("ChangedTest", aud["test_sentences_changed"])
mac("ChangedAll", aud["sentences_changed"])
mac("TokPerChanged", f"{aud['mean_tokens_changed_per_changed_sentence']:.1f}")
mac("NegLost", aud["negation_tokens_lost"]); mac("NegRecovered", aud["negation_tokens_recovered_by_normalization"])
mac("ErrRaw", err["errors_raw"]); mac("ErrNorm", err["errors_normalized"])
mac("Fixed", err["fixed_by_normalization"]); mac("Broken", err["broken_by_normalization"])
mac("BrokenNeg", err["broken_with_negation_in_sentence"])
cats = err["primary_category_of_raw_errors"]
for k, n in (("spelling variation", "CatSpell"), ("slang/abbreviation", "CatSlang"), ("negation", "CatNeg")):
    mac(n, cats.get(k, 0))
mac("CatOther", [v for k, v in cats.items() if k.startswith("other")][0])
A = rob[0]
mac("RobARaw", pct(A["raw"]["consistency"])); mac("RobANorm", pct(A["normalized"]["consistency"]))
mac("RobAN", A["n_variants"])
mac("WordRawF", f3(abl.loc["wordonly_raw", "test_macro_f1"])); mac("WordNormF", f3(abl.loc["wordonly_normalized", "test_macro_f1"]))
mac("WordRawAcc", f3(abl.loc["wordonly_raw", "test_accuracy"])); mac("WordNormAcc", f3(abl.loc["wordonly_normalized", "test_accuracy"]))
mac("ManualAudit", ("%d/%d preserved" % (aud.get("manual_audit_counts", {}).get("preserved", 0), aud["manual_audit_n"]))
    if "manual_audit_n" in aud else r"\textbf{\color{red}[pending]}")
with open(f"{OUT}/macros.tex", "w") as f:
    for k, v in m.items():
        f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")

# ---------------- table: dataset ----------------
labs = ["negative", "neutral", "positive"]
with open(f"{OUT}/tab_data.tex", "w") as f:
    f.write("\\begin{tabular}{lrrrr}\n\\toprule\nSplit & Neg. & Neu. & Pos. & Total\\\\\n\\midrule\n")
    for nme, d in (("Train", train), ("Dev", dev), ("Test", test)):
        c = d.label.value_counts()
        f.write(f"{nme} & {c['negative']:,} & {c['neutral']:,} & {c['positive']:,} & {len(d):,}\\\\\n")
    allc = pd.concat([train, dev, test]).label.value_counts()
    f.write(f"\\midrule\nAll & {allc['negative']:,} & {allc['neutral']:,} & {allc['positive']:,} & 20,000\\\\\n\\bottomrule\n\\end{{tabular}}\n")

# ---------------- table: main ----------------
with open(f"{OUT}/tab_main.tex", "w") as f:
    f.write("\\begin{tabular}{lcccc}\n\\toprule\nCondition & $C$ & Dev F1 & Test F1 & Test Acc\\\\\n\\midrule\n")
    for nme, r in (("Raw text", raw), ("Normalized", nrm)):
        f.write(f"{nme} & {r['best_C']} & {f3(r['dev_macro_f1'])} & {f4(r['test_macro_f1'])} & {f4(r['test_accuracy'])}\\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

# ---------------- table: ablation ----------------
names = {"negation": "Negation", "spelling": "Spelling", "abbreviation": "Abbreviation",
         "english_slang": "English slang", "elongation": "Elongation"}
def row(label, key):
    r = abl.loc[key]
    return f"{label} & {f3(r['dev_macro_f1'])} & {f3(r['test_macro_f1'])} & {f3(r['test_accuracy'])}\\\\\n"
with open(f"{OUT}/tab_ablation.tex", "w") as f:
    f.write("\\begin{tabular}{lccc}\n\\toprule\nConfiguration & Dev F1 & Test F1 & Test Acc\\\\\n\\midrule\n")
    f.write("\\multicolumn{4}{l}{\\emph{Word + char n-grams}}\\\\\n")
    f.write(row("Raw (no rules)", "full_raw"))
    for g, n in names.items():
        f.write(row(f"\\quad only {n}", f"only_{g}"))
    f.write(row("All groups", "full_normalized"))
    for g, n in names.items():
        f.write(row(f"\\quad $-$ {n}", f"without_{g}"))
    f.write("\\midrule\n\\multicolumn{4}{l}{\\emph{Word n-grams only}}\\\\\n")
    f.write(row("Raw (no rules)", "wordonly_raw"))
    for g, n in names.items():
        f.write(row(f"\\quad only {n}", f"wordonly_only_{g}"))
    f.write(row("All groups", "wordonly_normalized"))
    f.write("\\bottomrule\n\\end{tabular}\n")

# ---------------- table: robustness ----------------
with open(f"{OUT}/tab_robust.tex", "w") as f:
    f.write("\\begin{tabular}{lrcccc}\n\\toprule\nVariant set & $n$ & Raw & Norm. & Gain & 95\\% CI\\\\\n\\midrule\n")
    for r in rob:
        nm = "A: attested swap" if r["set"].startswith("A") else f"B: noise $p{{=}}{r['noise_p']}$"
        lo, hi = r["gain_95ci"]
        f.write(f"{nm} & {r['n_variants']:,} & {pct(r['raw']['consistency'])} & {pct(r['normalized']['consistency'])} & "
                f"{100*r['consistency_gain_norm_minus_raw']:+.1f} & [{100*lo:+.1f}, {100*hi:+.1f}]\\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

# ---------------- table: class-wise ----------------
with open(f"{OUT}/tab_classwise.tex", "w") as f:
    f.write("\\begin{tabular}{llccc}\n\\toprule\nClass & Cond. & P & R & F1\\\\\n\\midrule\n")
    for c in labs:
        for cond, short in (("raw", "raw"), ("normalized", "norm")):
            r = cw[(cw["class"] == c) & (cw.condition == cond)].iloc[0]
            lab = c.capitalize() if cond == "raw" else ""
            f.write(f"{lab} & {short} & {f3(r.precision)} & {f3(r.recall)} & {f3(r.f1)}\\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

# ---------------- table: error analysis ----------------
sub = err["subsets"]
with open(f"{OUT}/tab_subsets.tex", "w") as f:
    f.write("\\begin{tabular}{lrcc}\n\\toprule\nTest subset & $n$ & Raw F1 & Norm F1\\\\\n\\midrule\n")
    lab = {"changed_by_normalization": "Changed by rules", "unchanged": "Unchanged",
           "has_negation": "Contains negation", "has_spelling_variant": "Spelling variant",
           "has_slang_abbrev": "Slang / abbrev."}
    for k, v in lab.items():
        s = sub[k]
        f.write(f"{v} & {s['n']:,} & {f3(s['raw_macro_f1'])} & {f3(s['norm_macro_f1'])}\\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")
with open(f"{OUT}/tab_errors.tex", "w") as f:
    f.write("\\begin{tabular}{lrr}\n\\toprule\nPrimary category (raw errors) & Count & Share\\\\\n\\midrule\n")
    tot = sum(cats.values())
    for k, v in sorted(cats.items(), key=lambda x: -x[1]):
        k2 = k.replace("(sarcasm, context, neutral ambiguity)", "(context/sarcasm)")
        f.write(f"{k2.capitalize()} & {v} & {pct(v/tot)}\\%\\\\\n")
    f.write(f"\\midrule\nTotal raw errors & {tot} & 100\\%\\\\\n\\bottomrule\n\\end{{tabular}}\n")

# ---------------- rule table excerpt (appendix) ----------------
ra = pd.read_csv(f"{R}/rule_audit.csv").head(14)
esc = lambda s: str(s).replace("_", "\\_")
with open(f"{OUT}/tab_rules.tex", "w") as f:
    f.write("\\begin{tabular}{llr}\n\\toprule\nVariant $\\rightarrow$ canonical & Group & Freq.\\\\\n\\midrule\n")
    for r in ra.itertuples():
        f.write(f"{esc(r.variant)} $\\rightarrow$ {esc(r.canonical)} & {esc(r.group)} & {r.freq_all_splits:,}\\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

# ---------------- commit log table ----------------
if not os.path.exists(f"{OUT}/tab_commits.tex"):
    with open(f"{OUT}/tab_commits.tex", "w") as f:
        f.write("\\textbf{\\color{red}Run \\texttt{python src/commit\\_log.py} to generate the commit table.}\n")

# ---------------- extra macros (appended) ----------------
from normalize import GROUPS
ex = {}
for g, n in (("negation", "GrpNeg"), ("spelling", "GrpSpell"), ("abbreviation", "GrpAbbr"), ("english_slang", "GrpSlang")):
    ex[n] = sum(len(v) for v in GROUPS[g].values())
for r, tag in zip(rob[1:], ("BA", "BB", "BC", "BD")):
    ex[f"Rob{tag}Raw"] = pct(r["raw"]["consistency"]); ex[f"Rob{tag}Norm"] = pct(r["normalized"]["consistency"])
    ex[f"Rob{tag}Gain"] = f"{100*r['consistency_gain_norm_minus_raw']:+.1f}"
    lo, hi = r["gain_95ci"]; ex[f"Rob{tag}CI"] = f"[{100*lo:+.1f}, {100*hi:+.1f}]"
lo, hi = A["gain_95ci"]; ex["RobAGain"] = f"{100*A['consistency_gain_norm_minus_raw']:+.1f}"; ex["RobACI"] = f"[{100*lo:+.1f}, {100*hi:+.1f}]"
ex["RobAFRaw"] = f3(A["raw"]["macro_f1_variants"]); ex["RobAFNorm"] = f3(A["normalized"]["macro_f1_variants"])
ex["RobAFOrigRaw"] = f3(A["raw"]["macro_f1_originals_same_rows"]); ex["RobAFOrigNorm"] = f3(A["normalized"]["macro_f1_originals_same_rows"])
er, en = err["error_rate_with_flag_raw"], err["error_rate_without_flag_raw"]
ex["ErrSpellWith"] = pct(er["spelling_variant"]); ex["ErrSpellWithout"] = pct(en["spelling_variant"])
ex["ErrSlangWith"] = pct(er["slang_abbrev"]); ex["ErrSlangWithout"] = pct(en["slang_abbrev"])
ex["ErrNegWith"] = pct(er["negation"]); ex["ErrNegWithout"] = pct(en["negation"])
ex["NegTestN"] = err["subsets"]["has_negation"]["n"]
ex["ShareTrueNeutral"] = pct(err["share_of_raw_errors_with_true_neutral"], 0)
ex["ShareNeutralPred"] = pct(err["share_of_raw_errors_with_pred_neutral"], 0)
ex["SubChgRaw"] = f3(err["subsets"]["changed_by_normalization"]["raw_macro_f1"]); ex["SubChgNorm"] = f3(err["subsets"]["changed_by_normalization"]["norm_macro_f1"])
ex["SubUnchRaw"] = f3(err["subsets"]["unchanged"]["raw_macro_f1"]); ex["SubUnchNorm"] = f3(err["subsets"]["unchanged"]["norm_macro_f1"])
piv = pd.read_csv(f"{R}/classwise_delta.csv").set_index("class")
for c, n in (("negative", "DNeg"), ("neutral", "DNeu"), ("positive", "DPos")):
    ex[n] = f"{piv.loc[c, 'delta_f1']:+.3f}"
ex["AblMinF"] = f3(abl[abl.note != "word+char"].query("use_char_ngrams == True").test_macro_f1.min())
ex["AblMaxF"] = f3(abl[abl.note != "word+char"].query("use_char_ngrams == True").test_macro_f1.max())
for g, n in (("negation", "OnlyNeg"), ("spelling", "OnlySpell"), ("abbreviation", "OnlyAbbr"), ("english_slang", "OnlySlang"), ("elongation", "OnlyElong")):
    ex[f"Abl{n}F"] = f3(abl.loc[f"only_{g}", "test_macro_f1"])
    ex[f"AblW{n}F"] = f3(abl.loc[f"wordonly_only_{g}", "test_macro_f1"])
for k, v in aud.items():
    if k.startswith("sentences_changed_by_"):
        ex["Chg" + k.split("_by_")[1].title().replace("_", "")] = f"{v:,}"
with open(f"{OUT}/macros.tex", "a") as f:
    for k, v in ex.items():
        f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")


# ================= additional analyses =================
voc = J("vocab_stats.json")
lc = pd.read_csv(f"{R}/learning_curve.csv")
ch, un = err["changed_sentences"], err["unchanged_sentences"]
ex2 = {
    "VocabVarShare": f"{voc['variant_token_share_pct']:.1f}", "TypesRed": f"{voc['types_reduction_pct']:.1f}",
    "TypesRedTwo": f"{voc['types_df2_reduction_pct']:.1f}",
    "OOVRaw": f"{voc['raw']['test_oov_rate_df2_pct']:.1f}", "OOVNorm": f"{voc['normalized']['test_oov_rate_df2_pct']:.1f}",
    "TrainTypesRaw": f"{voc['raw']['train_types']:,}", "TrainTypesNorm": f"{voc['normalized']['train_types']:,}",
    "FixedChg": ch["fixed"], "BrokenChg": ch["broken"], "FlipsChg": ch["prediction_flips"],
    "FlipChgPct": f"{err['flip_pct_changed_sentences']:.1f}", "FlipUnchPct": f"{err['flip_pct_unchanged_sentences']:.1f}",
    "UnchN": f"{un['n']:,}", "NetChg": f"{ch['fixed']-ch['broken']:+d}".replace("-", "$-$"), "FlipsUnch": un["prediction_flips"], "ChurnRefPct": f"{err['churn_reference_raw_C025_vs_C05_flip_pct']:.1f}",
    "NegChgN": ch["with_negation"], "FlipsChgNeg": ch["flips_with_negation"], "FixedChgNeg": ch["fixed_with_negation"],
    "BrokenChgNeg": ch["broken_with_negation"],
    "FlipNegShare": f"{100*ch['flips_with_negation']/ch['prediction_flips']:.0f}",
    "NegChgShare": f"{100*ch['with_negation']/ch['n']:.0f}",
}
g = lc.groupby("fraction")
for fr, tag in ((0.1, "Ten"), (0.25, "Quarter"), (0.5, "Half"), (1.0, "Full")):
    d = lc[lc.fraction == fr]
    ex2[f"LC{tag}Diff"] = f"{d['diff'].mean():+.3f}"
    ex2[f"LC{tag}Raw"] = f3(d.raw.mean()); ex2[f"LC{tag}Norm"] = f3(d.normalized.mean())
    ex2[f"LC{tag}Pos"] = f"{int((d['diff'] > 0).sum())}/{len(d)}"
sa = A["subsets"]
ex2.update({"RobANegN": f"{sa['touches_negation_family']['n']:,}", "RobANegRaw": pct(sa["touches_negation_family"]["raw_consistency"]),
            "RobAOthN": f"{sa['other_families']['n']:,}", "RobAOthRaw": pct(sa["other_families"]["raw_consistency"])})
with open(f"{OUT}/macros.tex", "a") as f:
    for k, v in ex2.items():
        f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")

# vocabulary table
with open(f"{OUT}/tab_vocab.tex", "w") as f:
    f.write("\\begin{tabular}{lrr}\n\\toprule\n & Raw & Normalized\\\\\n\\midrule\n")
    f.write(f"Train word types & {voc['raw']['train_types']:,} & {voc['normalized']['train_types']:,}\\\\\n")
    f.write(f"Types with df$\\ge$2 & {voc['raw']['train_types_df2']:,} & {voc['normalized']['train_types_df2']:,}\\\\\n")
    f.write(f"Test OOV tokens (df$\\ge$2 vocab) & {voc['raw']['test_oov_rate_df2_pct']:.1f}\\% & {voc['normalized']['test_oov_rate_df2_pct']:.1f}\\%\\\\\n")
    f.write(f"Train tokens covered by rules & \\multicolumn{{2}}{{c}}{{{voc['variant_token_share_pct']:.1f}\\%}}\\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

# learning curve table
with open(f"{OUT}/tab_lc.tex", "w") as f:
    f.write("\\begin{tabular}{rrccc}\n\\toprule\nTrain frac. & $n$ & Raw F1 & Norm F1 & $\\Delta$ (mean$\\pm$sd)\\\\\n\\midrule\n")
    for fr, d in lc.groupby("fraction"):
        sd = d["diff"].std() if len(d) > 1 else float("nan")
        sdtxt = f"$\\pm${sd:.3f}" if sd == sd else ""
        f.write(f"{int(fr*100)}\\% & {int(d.n_train.iloc[0]):,} & {f3(d.raw.mean())} & {f3(d.normalized.mean())} & {d['diff'].mean():+.3f}{sdtxt}\\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

# C sensitivity (dev Macro-F1)
cs = list(raw["dev_macro_f1_by_C"].keys())
with open(f"{OUT}/tab_cdev.tex", "w") as f:
    f.write("\\begin{tabular}{l" + "c" * len(cs) + "}\n\\toprule\n$C$ & " + " & ".join(cs) + "\\\\\n\\midrule\n")
    f.write("Raw & " + " & ".join(f3(raw["dev_macro_f1_by_C"][c]) for c in cs) + "\\\\\n")
    f.write("Normalized & " + " & ".join(f3(nrm["dev_macro_f1_by_C"][c]) for c in cs) + "\\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

# rule-group summary
ex_var = {"negation": "nhi, nahin, nehi", "spelling": "acha, bohot, kyu, jese", "abbreviation": "bht, kr, rha, gya",
          "english_slang": "u, ur, plz, gud", "elongation": "nahiii, yayyyy"}
from normalize import GROUPS as _G
with open(f"{OUT}/tab_groups.tex", "w") as f:
    f.write("\\begin{tabular}{lrrl}\n\\toprule\nGroup & Rules & Sent. changed & Examples\\\\\n\\midrule\n")
    for gname, label in (("negation", "Negation"), ("spelling", "Spelling"), ("abbreviation", "Abbreviation"),
                         ("english_slang", "English slang"), ("elongation", "Elongation")):
        nr = sum(len(v) for v in _G[gname].values()) if gname in _G else "rule"
        nr = f"{nr}" if gname in _G else "regex"
        key = "sentences_changed_by_" + gname
        f.write(f"{label} & {nr} & {aud[key]:,} & \\emph{{{ex_var[gname]}}}\\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

# confusion matrices
labs3 = ["negative", "neutral", "positive"]
cm = {c: pd.read_csv(f"{R}/confusion_{c}.csv", index_col=0) for c in ("raw", "normalized")}
with open(f"{OUT}/tab_confusion.tex", "w") as f:
    f.write("\\begin{tabular}{l|ccc|ccc}\n\\toprule\n & \\multicolumn{3}{c|}{Raw (pred.)} & \\multicolumn{3}{c}{Normalized (pred.)}\\\\\nTrue & neg & neu & pos & neg & neu & pos\\\\\n\\midrule\n")
    for t in labs3:
        f.write(f"{t[:3]} & " + " & ".join(str(int(cm['raw'].loc[t, p])) for p in labs3) + " & " +
                " & ".join(str(int(cm['normalized'].loc[t, p])) for p in labs3) + "\\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

# qualitative examples (changed sentences only; ASCII-only, no profanity)
import re, random
ea = pd.read_csv(f"{R}/error_analysis_sample.csv")
bad = set("chutiya chutiye chutia bc mc gaand gandu bhadwe bhadwa madarchod behenchod saala saale sala harami kutta kutte kuty randi lund lauda bhosdi gand chut fuck shit bitch ass dick pig rape".split())
def okrow(r):
    t = str(r.text)
    return t.isascii() and not (set(re.findall(r"[a-z]+", t)) & bad) and 25 < len(t) < 120 and "<url>" not in t
def tex(sx):
    sx = sx.replace("<user>", "@u")
    for a, b in (("\\", ""), ("&", "\\&"), ("%", "\\%"), ("$", "\\$"), ("#", "\\#"), ("_", "\\_"), ("{", "\\{"), ("}", "\\}"), ("~", "\\~{}"), ("^", "")):
        sx = sx.replace(a, b)
    return re.sub(r"(@u\s*)+", "@u ", sx).strip()
rows = []
for case, lab in (("fixed_by_norm", "fixed"), ("broken_by_norm", "broken")):
    x = ea[(ea.case == case) & (ea.changed_by_norm)]
    x = x[x.apply(okrow, axis=1)].head(3)
    for r in x.itertuples():
        diff = ", ".join(f"{a}$\\to${b}" for a, b in zip(re.findall(r"[a-z]+", r.text), re.findall(r"[a-z]+", r.normalized)) if a != b)
        rows.append((lab, tex(r.text)[:90], diff.replace("_", "\\_"), r.true[:3], r.raw_pred[:3], r.norm_pred[:3]))
with open(f"{OUT}/tab_examples.tex", "w") as f:
    f.write("\\begin{tabular}{@{}l p{2.9cm} p{1.6cm} c c c@{}}\n\\toprule\n & Tweet (excerpt) & Rules & Gold & Raw & Norm\\\\\n\\midrule\n")
    for r in rows:
        f.write(f"{r[0]} & \\scriptsize {r[1]} & \\scriptsize {r[2]} & {r[3]} & {r[4]} & {r[5]}\\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")


# learning-curve figure
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(4.2, 2.6))
gm = lc.groupby("fraction")
xs = [int(f * 100) for f in gm.groups]
for col, lab, mk in (("raw", "raw", "o-"), ("normalized", "normalized", "s-")):
    ax.errorbar(xs, gm[col].mean().values, yerr=gm[col].std().fillna(0).values, fmt=mk, capsize=2, label=lab)
ax.set_xscale("log"); ax.set_xticks(xs); ax.set_xticklabels([f"{x}%" for x in xs])
ax.set_xlabel("training data used"); ax.set_ylabel("test Macro-F1"); ax.grid(alpha=.3); ax.legend()
plt.tight_layout(); plt.savefig(f"{R}/learning_curve.png", dpi=200); plt.close()

# variant examples (appendix)
rs = pd.read_csv(f"{R}/robustness_variant_sample.csv")
rows = []
for setname, lab in (("A", "A"), ("B_p0.3", "B")):
    x = rs[rs.set == setname]
    x = x[x.apply(lambda r: str(r.original).isascii() and not (set(re.findall(r"[a-z]+", str(r.original))) & bad)
                  and 20 < len(str(r.original)) < 100 and "<url>" not in str(r.original), axis=1)].head(4)
    for r in x.itertuples():
        diff = ", ".join(f"{a}$\\to${b}" for a, b in zip(re.findall(r"[a-z]+", r.original), re.findall(r"[a-z]+", r.variant)) if a != b)
        rows.append((lab, tex(r.original)[:80], diff.replace("_", "\\_")[:70]))
with open(f"{OUT}/tab_variants.tex", "w") as f:
    f.write("\\begin{tabular}{@{}l p{3.5cm} p{3.2cm}@{}}\n\\toprule\nSet & Original (excerpt) & Token changes\\\\\n\\midrule\n")
    for r in rows:
        f.write(f"{r[0]} & \\scriptsize {r[1]} & \\scriptsize {r[2]}\\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")


# audit statistics table (appendix)
with open(f"{OUT}/tab_audit.tex", "w") as f:
    f.write("\\begin{tabular}{lr}\n\\toprule\nQuantity & Value\\\\\n\\midrule\n")
    for lab, val in (("Rules (variant forms)", aud["rules_total"]), ("Rules observed in data", aud["rules_observed_in_data"]),
                     ("Rules violating negation compatibility", aud["rules_negation_incompatible"]),
                     ("Sentences (all splits)", f"{aud['sentences_total']:,}"), ("Sentences changed", f"{aud['sentences_changed']:,}"),
                     ("\\quad of which in test", f"{aud['test_sentences_changed']:,}"),
                     ("Changed by negation group", f"{aud['sentences_changed_by_negation']:,}"),
                     ("Changed by spelling group", f"{aud['sentences_changed_by_spelling']:,}"),
                     ("Changed by abbreviation group", f"{aud['sentences_changed_by_abbreviation']:,}"),
                     ("Changed by English-slang group", f"{aud['sentences_changed_by_english_slang']:,}"),
                     ("Changed by elongation", f"{aud['sentences_changed_by_elongation']:,}"),
                     ("Negation count unchanged", f"{aud['negation_count_unchanged']:,}"),
                     ("Negation tokens lost", aud["negation_tokens_lost"]),
                     ("Negation tokens recovered", aud["negation_tokens_recovered_by_normalization"])):
        f.write(f"{lab} & {val}\\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

print("report tables written to", OUT)
