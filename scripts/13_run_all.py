"""
Run the full pipeline end-to-end (steps 1 to 12).

  python scripts/13_run_all.py                 # all steps
  python scripts/13_run_all.py --skip 5 8      # skip listed steps
  python scripts/13_run_all.py --from 7        # start from step 7
  python scripts/13_run_all.py --to 9          # stop after step 9
  python scripts/13_run_all.py --continue-on-error
  python scripts/13_run_all.py --dry-run
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


# (step_number, script_filename, default_args, short_description, est_minutes)
STEPS = [
    (1,  "01_data_load.py",            [],                "Load CSV + OCR texts → corpus_joined", 2),
    (2,  "02_data_quality.py",         [],                "OCR quality filter + lexical diagnostics", 2),
    (3,  "03_preprocessing.py",        [],                "Light cleaning + lemma + stopwords", 8),
    (4,  "04_eda.py",                  [],                "EDA on text_clean: top words + keyword trends", 1),
    (5,  "05_chunking_eval.py",        [],                "Compare chunking methods (eval only)", 1),
    (6,  "06_chunking_embedding.py",   [],                "Chunk + embed (MiniLM, MPS/CUDA)", 8),
    (7,  "07_bertopic.py",             [],                "BERTopic + reduce_topics(30) (noise -1 kept)", 25),
    (8,  "08_lda.py",                  [],                "LDA (10 topics)", 8),
    (9,  "09_doc_topic_vectors.py",    [],                "Aggregate to doc level + party_family", 1),
    (10, "10_analyses.py",             ["--all"],         "Clustering + specialisation analyses", 2),
    (11, "11_visualizations.py",       ["--all"],         "Projection + sanity + native viz", 5),
    (12, "12_sentiment.py",            [],                "Sentiment scoring on shared topics", 30),
]


def run_step(step_num: int, script: str, args: list, description: str,
             continue_on_error: bool, dry_run: bool) -> tuple[bool, float]:
    print(f"\n{'='*78}")
    print(f"STEP {step_num:>2}  |  {script}")
    print(f"        {description}")
    print(f"{'='*78}")

    cmd = [sys.executable, str(SCRIPTS_DIR / script)] + args
    print(f"$ {' '.join(cmd)}\n")

    if dry_run:
        return True, 0.0

    start = time.time()
    try:
        result = subprocess.run(cmd, cwd=PROJECT_ROOT)
        elapsed = time.time() - start
        success = result.returncode == 0
        if not success:
            print(f"\n[STEP {step_num}] FAILED with exit code {result.returncode}")
            if not continue_on_error:
                return False, elapsed
        return success, elapsed
    except KeyboardInterrupt:
        print(f"\n[STEP {step_num}] Interrupted by user.")
        return False, time.time() - start


def fmt_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.1f}min"
    return f"{minutes / 60:.1f}h"


def main():
    parser = argparse.ArgumentParser(description="Run the full pipeline end-to-end.")
    parser.add_argument("--from", dest="from_step", type=int, default=1,
                        help="Start from this step number (default: 1).")
    parser.add_argument("--to", dest="to_step", type=int, default=12,
                        help="Stop after this step number (default: 12).")
    parser.add_argument("--skip", type=int, nargs="*", default=[],
                        help="Step numbers to skip (e.g. --skip 5 8).")
    parser.add_argument("--continue-on-error", action="store_true",
                        help="Don't stop the pipeline if a step fails.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the steps that would run without executing them.")
    args = parser.parse_args()

    selected = [s for s in STEPS
                if args.from_step <= s[0] <= args.to_step
                and s[0] not in args.skip]

    if not selected:
        print("No steps selected. Adjust --from / --to / --skip.")
        sys.exit(1)

    total_estimated = sum(s[4] for s in selected)
    print(f"Pipeline: {len(selected)} step(s)")
    print(f"Estimated total runtime: ~{total_estimated} min (your hardware may differ)")
    print(f"Skipped: {sorted(args.skip) if args.skip else '(none)'}")
    print(f"Range: step {args.from_step} → step {args.to_step}")
    if args.dry_run:
        print(f"\nDRY RUN — no execution.\n")

    results = []
    grand_start = time.time()

    for step_num, script, default_args, description, _ in selected:
        success, elapsed = run_step(
            step_num, script, default_args, description,
            args.continue_on_error, args.dry_run,
        )
        results.append((step_num, script, success, elapsed))
        if not success and not args.continue_on_error:
            break

    grand_total = time.time() - grand_start

    # Summary
    print(f"\n{'='*78}")
    print("PIPELINE SUMMARY")
    print(f"{'='*78}")
    for step_num, script, success, elapsed in results:
        status = "OK " if success else "FAIL"
        print(f"  [{status}] step {step_num:>2}  {script:<35}  {fmt_time(elapsed)}")
    print(f"\nTotal: {fmt_time(grand_total)}")

    n_fail = sum(1 for _, _, s, _ in results if not s)
    if n_fail > 0:
        print(f"FAILED: {n_fail} step(s)")
        sys.exit(1)
    else:
        print("All steps completed successfully.")


if __name__ == "__main__":
    main()
