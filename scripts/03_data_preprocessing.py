"""
Step 3 — Text Cleaning and Preprocessing

Input:
data/corpus_cleaned.csv
data/stop_word_fr.txt

Output:
data/corpus_preprocessed.csv
outputs/preprocessing_info.txt
"""

from pathlib import Path
import re
import pandas as pd


# =====================
# Paths
# =====================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

INPUT_PATH = DATA_DIR / "corpus_cleaned.csv"
STOPWORDS_PATH = PROJECT_ROOT / "stop_word_fr.txt"

OUTPUT_PATH = DATA_DIR / "corpus_preprocessed.csv"
INFO_PATH = OUTPUT_DIR / "preprocessing_info.txt"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =====================
# Stopwords
# =====================

POLITICAL_WORDS_TO_KEEP = {
    "droite", "gauche",
    "etat", "état",
    "france", "français", "française",
    "nation", "national", "nationale",
    "république", "republique",
    "politique", "parti", "élection", "election",
    "candidat", "candidate", "vote",
    "travail", "travailleurs", "ouvriers",
    "emploi", "chômage", "chomage",
    "social", "économie", "economie",
    "sécurité", "securite",
    "immigration", "europe",
    "école", "ecole",
    "famille", "impôt", "impot",
    "liberté", "liberte",
    "justice", "réforme", "reforme",
    "pouvoir", "peuple",
    "agriculture", "entreprise",
    "salaire", "logement", "santé", "sante",
    "jeunesse", "femme", "femmes",
    "ouvrière", "ouvriere"
}


def load_stopwords(path: Path) -> set[str]:
    if not path.exists():
        raise FileNotFoundError(f"Stopword file not found: {path}")

    stopwords = set()

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            word = line.strip().lower()
            if word:
                stopwords.add(word)

    # Important: keep politically meaningful words
    stopwords = stopwords - POLITICAL_WORDS_TO_KEEP

    return stopwords


STOPWORDS = load_stopwords(STOPWORDS_PATH)


# =====================
# Cleaning functions
# =====================

def remove_recurrent_noise(text: str) -> str:
    if not isinstance(text, str):
        return ""

    text = text.lower()

    patterns = [
        r"sciences\s+po\s*/\s*fonds\s+cevipof",
        r"république\s+française\s*[-–—]?\s*liberté\s*[-–—]?\s*égalité\s*[-–—]?\s*fraternité",
        r"republique\s+francaise\s*[-–—]?\s*liberte\s*[-–—]?\s*egalite\s*[-–—]?\s*fraternite",
        r"\bimprimerie\b[^\n]{0,120}",
        r"\boffset\b[^\n]{0,120}",
        r"\batelier\s+d['’]arts\s+graphiques\b[^\n]{0,120}",
    ]

    for pattern in patterns:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)

    text = re.sub(r"[☐☒□■▪▫◻◼●○◆◇]", " ", text)

    return text


def remove_generic_campaign_phrases(text: str) -> str:
    if not isinstance(text, str):
        return ""

    patterns = [
        r"\bvotez\s+pour\b",
        r"\bje\s+vous\s+demande\b",
        r"\bje\s+sollicite\s+vos\s+suffrages\b",
        r"\bchers\s+électeurs\b",
        r"\bchers\s+electeurs\b",
        r"\bmes\s+chers\s+concitoyens\b",
        r"\bmadame\s+monsieur\b",
        r"\bmadame\s+mademoiselle\s+monsieur\b",
        r"\bvu\s+le\s+candidat\b",
        r"\bvu\s+la\s+candidate\b",
        r"\bvu\s+les\s+candidats\b",
        r"\ble\s+candidat\b",
        r"\bla\s+candidate\b",
        r"\ble\s+suppléant\b",
        r"\ble\s+suppleant\b",
    ]

    for pattern in patterns:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)

    return text


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    text = re.sub(r"[’ʼ`´]", "'", text)
    text = re.sub(r"[–—−]", "-", text)

    # Keep French letters and spaces
    text = re.sub(r"[^a-zàâäéèêëîïôöùûüçœæ\s'-]", " ", text)

    # Remove isolated one-letter OCR noise, except useful French clitics
    text = re.sub(r"\b(?![ldjtmnsqc])\w\b", " ", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def remove_stopwords(text: str) -> str:
    if not isinstance(text, str):
        return ""

    tokens = text.split()

    filtered = [
        token for token in tokens
        if token not in STOPWORDS and len(token) > 1
    ]

    return " ".join(filtered)


def preprocess_text(text: str) -> str:
    text = remove_recurrent_noise(text)
    text = remove_generic_campaign_phrases(text)
    text = normalize_text(text)
    text = remove_stopwords(text)
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

    print(f"Stopwords loaded after protection: {len(STOPWORDS)}")

    print("Preprocessing texts...")
    df["text_preprocessed"] = df["text"].apply(preprocess_text)

    df["tokens_before"] = df["text"].apply(
        lambda x: len(str(x).split()) if pd.notna(x) else 0
    )

    df["tokens_after"] = df["text_preprocessed"].apply(
        lambda x: len(x.split()) if isinstance(x, str) else 0
    )

    # Keep documents that still contain enough content after preprocessing
    df["preprocess_keep"] = df["tokens_after"] >= 30
    df_final = df[df["preprocess_keep"]].copy()

    with open(INFO_PATH, "w", encoding="utf-8") as f:
        f.write("=== PREPROCESSING INFO ===\n\n")
        f.write(f"Initial documents: {len(df)}\n")
        f.write(f"Documents kept after preprocessing: {len(df_final)}\n")
        f.write(f"Documents removed after preprocessing: {len(df) - len(df_final)}\n")
        f.write(f"Stopwords used: {len(STOPWORDS)}\n\n")

        f.write("=== TOKEN COUNTS ===\n\n")
        f.write(f"Average tokens before: {df['tokens_before'].mean():.2f}\n")
        f.write(f"Average tokens after: {df['tokens_after'].mean():.2f}\n")

        if df["tokens_before"].mean() > 0:
            reduction = 100 * (
                1 - df["tokens_after"].mean() / df["tokens_before"].mean()
            )
            f.write(f"Average reduction: {reduction:.2f}%\n\n")

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

        f.write("=== POLITICAL WORDS EXPLICITLY KEPT ===\n\n")
        f.write(", ".join(sorted(POLITICAL_WORDS_TO_KEEP)))
        f.write("\n")

    df_final.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print("Done.")
    print(f"Saved corpus: {OUTPUT_PATH}")
    print(f"Saved info: {INFO_PATH}")


if __name__ == "__main__":
    main()