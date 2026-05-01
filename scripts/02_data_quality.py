"""
Step 2 — OCR quality scoring and filter on the joined corpus.
"""

import re
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = PROJECT_ROOT / "outputs"

PREV_DIR = OUTPUTS / "01_data_load"
STEP_DIR = OUTPUTS / "02_data_quality"
FIG_DIR = STEP_DIR / "figures"
REPORTS_DIR = STEP_DIR / "reports"

STEP_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

INPUT_PATH = PREV_DIR / "corpus_joined.csv"
OUTPUT_PATH = STEP_DIR / "corpus_cleaned.csv"
STATS_PATH = REPORTS_DIR / "ocr_quality.txt"


COMMON_WORDS = set([
    "le", "la", "les", "de", "des", "du", "un", "une", "et", "en", "à", "pour",
    "avec", "dans", "sur", "plus", "mais", "ou", "donc", "car", "ni", "si",
    "est", "sont", "être", "avoir", "fait", "par", "comme", "même",
    "france", "politique", "élection", "candidat", "vote", "république",
    "travail", "emploi", "social", "économie", "sécurité", "nation"
])


def extract_tokens(text):
    if not isinstance(text, str):
        return []
    return re.findall(r"[a-zàâäéèêëîïôöùûüçœæ]+", text.lower())


def evaluate_text(text):
    tokens = extract_tokens(text)
    n = len(tokens)
    if n == 0:
        return {"tokens": 0, "known_ratio": 0, "short_ratio": 1, "score": 0}

    known_ratio = sum(t in COMMON_WORDS for t in tokens) / n
    short_ratio = sum(len(t) <= 2 for t in tokens) / n
    score = 0.7 * known_ratio + 0.3 * (1 - short_ratio)
    return {
        "tokens": n,
        "known_ratio": known_ratio,
        "short_ratio": short_ratio,
        "score": score,
    }


def make_quality_plots(df):
    plt.figure(figsize=(8, 5))
    df.boxplot(column="score", by="year", grid=False)
    plt.title("OCR quality score by year")
    plt.suptitle("")
    plt.xlabel("Year")
    plt.ylabel("Score")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "ocr_score_by_year.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    df["score"].hist(bins=40)
    plt.title("OCR score distribution")
    plt.xlabel("Score")
    plt.ylabel("Documents")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "ocr_score_distribution.png", dpi=150)
    plt.close()

    keep_by_year = df.groupby(["year", "keep"]).size().unstack(fill_value=0)
    keep_by_year.plot(kind="bar", stacked=True, figsize=(8, 5))
    plt.title("Kept vs removed documents")
    plt.xlabel("Year")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "kept_removed_by_year.png", dpi=150)
    plt.close()


def main():
    print("Loading dataset...")
    df = pd.read_csv(INPUT_PATH)

    if "text" not in df.columns:
        raise ValueError("Missing column: text")

    print("Computing OCR quality...")
    metrics = df["text"].apply(evaluate_text).apply(pd.Series)
    df = pd.concat([df, metrics], axis=1)

    df["keep"] = (df["tokens"] > 50) & (df["short_ratio"] < 0.5)
    df_clean = df[df["keep"]].copy()

    with open(STATS_PATH, "w", encoding="utf-8") as f:
        f.write("=== OCR QUALITY ===\n\n")
        f.write(f"Total documents: {len(df)}\n")
        f.write(f"Kept documents: {len(df_clean)}\n")
        f.write(f"Removed: {len(df) - len(df_clean)}\n\n")

        f.write("Average score by year:\n")
        f.write(str(df.groupby("year")["score"].mean()))
        f.write("\n\n")

        f.write("Median tokens by year:\n")
        f.write(str(df.groupby("year")["tokens"].median()))
        f.write("\n")

    print("Generating OCR quality plots...")
    make_quality_plots(df)

    df_clean.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print("Done.")
    print(f"Saved dataset → {OUTPUT_PATH}")
    print(f"Saved stats → {STATS_PATH}")
    print(f"Saved figures → {FIG_DIR}")


if __name__ == "__main__":
    main()
