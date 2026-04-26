"""
Step 3 — Text Cleaning and Preprocessing

Goal:
Clean OCR artifacts and prepare manifesto texts for topic modeling.

Input:
data/corpus_cleaned.csv

Output:
data/corpus_preprocessed.csv
outputs/preprocessing_info.txt
"""

from pathlib import Path
import re
import pandas as pd


# =====================
# Project paths
# =====================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

INPUT_PATH = DATA_DIR / "corpus_cleaned.csv"
OUTPUT_PATH = DATA_DIR / "corpus_preprocessed.csv"
INFO_PATH = OUTPUT_DIR / "preprocessing_info.txt"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =====================
# Cleaning functions
# =====================

def remove_recurrent_noise(text: str) -> str:
    """
    Remove recurrent OCR and archive-related noise that should not
    influence topic modeling.
    """
    if not isinstance(text, str):
        return ""

    patterns = [
        r"sciences\s+po\s*/\s*fonds\s+cevipof",
        r"république\s+française\s*[-–—]?\s*liberté\s*[-–—]?\s*égalité\s*[-–—]?\s*fraternité",
        r"republique\s+francaise\s*[-–—]?\s*liberte\s*[-–—]?\s*egalite\s*[-–—]?\s*fraternite",
        r"\bimprimerie\b[^\n]{0,120}",
        r"\boffset\b[^\n]{0,120}",
        r"\batelier\s+d['’]arts\s+graphiques\b[^\n]{0,120}",
    ]

    text = text.lower()

    for pattern in patterns:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)

    # Remove checkbox and OCR layout symbols
    text = re.sub(r"[☐☒□■▪▫◻◼●○◆◇]", " ", text)

    return text


def normalize_text(text: str) -> str:
    """
    Normalize text while preserving meaningful French characters.
    """
    if not isinstance(text, str):
        return ""

    # Normalize apostrophes and dashes
    text = re.sub(r"[’ʼ`´]", "'", text)
    text = re.sub(r"[–—−]", "-", text)

    # Remove numbers and punctuation, keep French letters, apostrophes and spaces
    text = re.sub(r"[^a-zàâäéèêëîïôöùûüçœæ\s'-]", " ", text)

    # Remove isolated one-character OCR noise except useful French clitics
    text = re.sub(r"\b(?![ldjtmnsqc])\w\b", " ", text)

    # Normalize spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def preprocess_text(text: str) -> str:
    text = remove_recurrent_noise(text)
    text = normalize_text(text)
    return text


# =====================
# Main
# =====================

def main():
    print("Loading corpus...")
    df = pd.read_csv(INPUT_PATH)
    print(f"Documents loaded: {len(df)}")

    if "text" not in df.columns:
        raise ValueError("Column 'text' is missing from the input file.")

    print("Preprocessing texts...")
    df["text_preprocessed"] = df["text"].apply(preprocess_text)

    df["tokens_before"] = df["text"].apply(
        lambda x: len(str(x).split()) if pd.notna(x) else 0
    )

    df["tokens_after"] = df["text_preprocessed"].apply(
        lambda x: len(x.split()) if isinstance(x, str) else 0
    )

    # Optional safety filter after preprocessing
    df["preprocess_keep"] = df["tokens_after"] >= 50

    df_final = df[df["preprocess_keep"]].copy()

    # =====================
    # Save information
    # =====================

    with open(INFO_PATH, "w", encoding="utf-8") as f:
        f.write("=== PREPROCESSING INFO ===\n\n")

        f.write(f"Initial documents: {len(df)}\n")
        f.write(f"Documents kept after preprocessing: {len(df_final)}\n")
        f.write(f"Documents removed after preprocessing: {len(df) - len(df_final)}\n\n")

        f.write("=== TOKEN COUNTS ===\n\n")
        f.write(f"Average tokens before: {df['tokens_before'].mean():.2f}\n")
        f.write(f"Average tokens after: {df['tokens_after'].mean():.2f}\n")
        f.write(
            f"Average reduction: "
            f"{100 * (1 - df['tokens_after'].mean() / df['tokens_before'].mean()):.2f}%\n\n"
        )

        f.write("Token distribution after preprocessing:\n")
        f.write(df["tokens_after"].describe().to_string())
        f.write("\n\n")

        if "year" in df.columns:
            f.write("=== TOKEN DISTRIBUTION BY YEAR ===\n\n")
            f.write(
                df.groupby("year")["tokens_after"]
                .describe()[["count", "mean", "min", "50%", "max"]]
                .to_string()
            )
            f.write("\n\n")

    # =====================
    # Save corpus
    # =====================

    df_final.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print("Done.")
    print(f"Saved corpus: {OUTPUT_PATH}")
    print(f"Saved info: {INFO_PATH}")


if __name__ == "__main__":
    main()