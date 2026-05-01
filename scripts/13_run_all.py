"""
Run the full pipeline end-to-end.

Just: python scripts/13_run_all.py
"""

import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

STEPS = [
    "01_data_load.py",
    "02_data_quality.py",
    "03_preprocessing.py",
    "04_eda.py",
    "05_chunking_eval.py",
    "06_chunking_embedding.py",
    "07_bertopic.py",
    "09_doc_topic_vectors.py",
    "10_analyses.py",
    "11_visualizations.py",
    "12_sentiment.py",
]


def fmt_time(seconds):
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}min"
    return f"{seconds / 3600:.1f}h"


def main():
    grand_start = time.time()
    timings = []

    for step, script in enumerate(STEPS, start=1):
        print(f"\n{'=' * 78}\nSTEP {step:>2}  |  {script}\n{'=' * 78}\n")
        start = time.time()
        result = subprocess.run([sys.executable, str(SCRIPTS_DIR / script)], cwd=PROJECT_ROOT)
        elapsed = time.time() - start
        timings.append((step, script, result.returncode == 0, elapsed))
        if result.returncode != 0:
            print(f"\n[STEP {step}] FAILED")
            sys.exit(1)

    print(f"\n{'=' * 78}\nPIPELINE SUMMARY\n{'=' * 78}")
    for step, script, ok, elapsed in timings:
        print(f"  [{'OK' if ok else 'FAIL'}] step {step:>2}  {script:<35}  {fmt_time(elapsed)}")
    print(f"\nTotal: {fmt_time(time.time() - grand_start)}")


if __name__ == "__main__":
    main()
