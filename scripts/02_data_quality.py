"""
Step 2 — OCR Quality Assessment (clean + plots)

Input:
data/corpus_joined.csv

Outputs:
data/corpus_cleaned.csv
outputs/ocr_quality.txt
figures/*.png
"""

from pathlib import Path
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt


# =====================
# Paths
# =====================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIG_DIR = PROJECT_ROOT / "figures"

INPUT_PATH = DATA_DIR / "corpus_joined.csv"
OUTPUT_PATH = DATA_DIR / "corpus_cleaned.csv"
STATS_PATH = OUTPUT_DIR / "ocr_quality.txt"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


# =====================
# Small French vocabulary
# =====================

COMMON_WORDS = set([
    "le","la","les","de","des","du","un","une","et","en","à","pour",
    "avec","dans","sur","plus","mais","ou","donc","car","ni","si",
    "est","sont","être","avoir","fait","par","comme","même",
    "france","politique","élection","candidat","vote","république",
    "travail","emploi","social","économie","sécurité","nation"
])


# =====================
# OCR quality function
# =====================

def evaluate_text(text):
    if not isinstance(text, str) or len(text) < 20:
        return {"tokens": 0, "known_ratio": 0, "short_ratio": 1, "score": 0}

    tokens = re.findall(r"[a-zàâéèêëîïôûùç]+", text.lower())
    n = len(tokens)

    if n == 0:
        return {"tokens": 0, "known_ratio": 0, "short_ratio": 1, "score": 0}

    known = sum(t in COMMON_WORDS for t in tokens) / n
    short = sum(len(t) <= 2 for t in tokens) / n

    score = 0.7 * known + 0.3 * (1 - short)

    return {
        "tokens": n,
        "known_ratio": known,
        "short_ratio": short,
        "score": score
    }


# =====================
# Plots
# =====================

def make_quality_plots(df):

    # Score OCR par année
    plt.figure(figsize=(8, 5))
    df.boxplot(column="score", by="year", grid=False)
    plt.title("OCR quality score by year")
    plt.suptitle("")
    plt.xlabel("Year")
    plt.ylabel("Score")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "ocr_score_by_year.png", dpi=150)
    plt.close()

    # Longueur texte
    plt.figure(figsize=(8, 5))
    df.boxplot(column="tokens", by="year", grid=False, showfliers=False)
    plt.title("Document length by year")
    plt.suptitle("")
    plt.xlabel("Year")
    plt.ylabel("Tokens")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "text_length_by_year.png", dpi=150)
    plt.close()

    # Distribution score
    plt.figure(figsize=(8, 5))
    df["score"].hist(bins=40)
    plt.title("OCR score distribution")
    plt.xlabel("Score")
    plt.ylabel("Documents")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "ocr_score_distribution.png", dpi=150)
    plt.close()

    # Gardé vs supprimé
    keep_by_year = df.groupby(["year", "keep"]).size().unstack(fill_value=0)

    keep_by_year.plot(kind="bar", stacked=True, figsize=(8, 5))
    plt.title("Kept vs removed documents")
    plt.xlabel("Year")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "kept_removed_by_year.png", dpi=150)
    plt.close()


# =====================
# Main
# =====================

def main():

    print("Loading dataset...")
    df = pd.read_csv(INPUT_PATH)

    print("Computing OCR quality...")
    metrics = df["text"].apply(evaluate_text).apply(pd.Series)
    df = pd.concat([df, metrics], axis=1)

    # =====================
    # Filtering
    # =====================

    df["keep"] = (
        (df["tokens"] > 50) &
        (df["short_ratio"] < 0.5)
    )

    df_clean = df[df["keep"]].copy()

    # =====================
    # Stats
    # =====================

    with open(STATS_PATH, "w") as f:
        f.write("=== OCR QUALITY ===\n\n")
        f.write(f"Total documents: {len(df)}\n")
        f.write(f"Kept documents: {len(df_clean)}\n")
        f.write(f"Removed: {len(df) - len(df_clean)}\n\n")

        f.write("Average score:\n")
        f.write(str(df.groupby("year")["score"].mean()))
        f.write("\n\n")

        f.write("Median length:\n")
        f.write(str(df.groupby("year")["tokens"].median()))

    # =====================
    # Plots
    # =====================

    print("Generating plots...")
    make_quality_plots(df)

    # =====================
    # Save cleaned dataset
    # =====================

    df_clean.to_csv(OUTPUT_PATH, index=False)

    print("Done.")
    print(f"Saved dataset → {OUTPUT_PATH}")
    print(f"Saved stats → {STATS_PATH}")
    print(f"Saved figures → {FIG_DIR}")


if __name__ == "__main__":
    main()