"""Write commit hashes + timestamps to results/commit_log.txt and a LaTeX table.

Usage (from the project root, after committing):
    python src/commit_log.py
The table lists every commit up to HEAD. A commit cannot contain its own hash, so the
commit that stores the regenerated table is, by construction, the one entry it cannot list;
the full, current history is always visible with `git log` and on GitHub.
"""
import subprocess, os

fmt = "%h|%H|%ad|%an|%s"
out = subprocess.run(["git", "log", "--reverse", f"--pretty=format:{fmt}", "--date=iso"],
                     capture_output=True, text=True, check=True).stdout.strip().splitlines()
os.makedirs("results", exist_ok=True)
os.makedirs("report/gen", exist_ok=True)

with open("results/commit_log.txt", "w", encoding="utf-8", newline="\n") as f:
    f.write("short_hash | full_hash | timestamp | author | message\n")
    for l in out:
        f.write(l.replace("|", " | ") + "\n")


def esc(s):
    for a, b in [("\\", "\\textbackslash{}"), ("_", "\\_"), ("&", "\\&"), ("%", "\\%"),
                 ("#", "\\#"), ("$", "\\$"), ("{", "\\{"), ("}", "\\}")]:
        s = s.replace(a, b)
    return s


with open("report/gen/tab_commits.tex", "w", encoding="utf-8", newline="\n") as f:
    # message column wraps instead of being cut off mid-word
    f.write("\\scriptsize\n\\begin{tabular}{@{}l l p{1.6in}@{}}\n\\toprule\n"
            "Hash & Timestamp (IST) & Message\\\\\n\\midrule\n")
    for l in out:
        h, H, d, a, s = l.split("|", 4)
        f.write(f"\\texttt{{{h}}} & {d[:16]} & {esc(s)}\\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")
print(f"{len(out)} commits written")
