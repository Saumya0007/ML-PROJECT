"""Reproduce every result in the report (CPU only, ~10-15 min).
Usage:  python run_all.py            (from the project root, with data/ filled)"""
import subprocess, sys, os, time

os.makedirs("results", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)
steps = [
    "src/load_data.py",          # sanity check: 14000 3000 3000 + class counts
    "src/normalize.py",          # rule-table self-check
    "src/baseline.py",           # raw vs normalized -> results/*.json, checkpoints/*.joblib
    "src/vocab_stats.py",         # vocabulary / OOV statistics
    "src/ablation.py",           # rule-group + word-only ablations
    "src/learning_curve.py",     # data-efficiency experiment
    "src/significance.py",       # McNemar + paired bootstrap
    "src/robustness.py",         # consistency on spelling variants
    "src/audit.py",              # sentiment-preservation audit
    "src/analysis.py",           # class-wise + error analysis
    "src/make_report_tables.py", # LaTeX tables/macros for the report
    "src/othermodels.py",       # raw vs normalized for other classifiers
    ]
t0 = time.time()
for s in steps:
    print(f"\n=== {s} ===", flush=True)
    subprocess.run([sys.executable, s], check=True,
                   env={**os.environ, "PYTHONPATH": "src", "PYTHONHASHSEED": "0"})
print(f"\nAll steps finished in {(time.time()-t0)/60:.1f} min. See results/ and report/gen/.")
