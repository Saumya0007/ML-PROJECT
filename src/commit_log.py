"""Write commit hashes + timestamps to results/commit_log.txt and a LaTeX table."""
import subprocess, os

fmt = "%h|%H|%ad|%an|%s"
out = subprocess.run(["git", "log", "--reverse", f"--pretty=format:{fmt}", "--date=iso"],
                     capture_output=True, text=True, check=True).stdout.strip().splitlines()
os.makedirs("results", exist_ok=True)
os.makedirs("report/gen", exist_ok=True)
with open("results/commit_log.txt", "w", encoding="utf-8") as f:
    f.write("short_hash | full_hash | timestamp | author | message\n")
    for l in out:
        f.write(l.replace("|", " | ") + "\n")
esc = lambda s: s.replace("_", "\\_").replace("&", "\\&").replace("%", "\\%").replace("#", "\\#")
with open("report/gen/tab_commits.tex", "w", encoding="utf-8") as f:
    f.write("\\begin{tabular}{@{}lll@{}}\n\\toprule\nHash & Timestamp & Message\\\\\n\\midrule\n")
    for l in out:
        h, H, d, a, s = l.split("|", 4)
        f.write(f"\\texttt{{{h}}} & {d[:16]} & {esc(s[:48])}\\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")
print(f"{len(out)} commits written")
