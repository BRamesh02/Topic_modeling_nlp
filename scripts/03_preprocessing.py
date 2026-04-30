"""
Step 3 — Two-track preprocessing.

  text_clean         lightly cleaned, kept for the sentence-transformer.
  text_preprocessed  lemmatised + stopwords removed, used by LDA and c-TF-IDF.

Order: normalize -> lemmatize -> remove_stopwords (so stopwords are matched on
the lemma form).
"""

from pathlib import Path
import re
import numpy as np
import pandas as pd
import spacy
from tqdm import tqdm

nlp = spacy.load("fr_core_news_sm", disable=["parser", "ner"])


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = PROJECT_ROOT / "outputs"

PREV_DIR = OUTPUTS / "02_data_quality"
STEP_DIR = OUTPUTS / "03_preprocessing"
REPORTS_DIR = STEP_DIR / "reports"
FIG_DIR = STEP_DIR / "figures"

STEP_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

INPUT_PATH = PREV_DIR / "corpus_cleaned.csv"
STOPWORDS_PATH = PROJECT_ROOT / "stop_word_fr.txt"

OUTPUT_PATH = STEP_DIR / "corpus_preprocessed.csv"
INFO_PATH = REPORTS_DIR / "preprocessing_info.txt"


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

    stopwords = stopwords - POLITICAL_WORDS_TO_KEEP
    return stopwords


STOPWORDS = load_stopwords(STOPWORDS_PATH)


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
        r"\batelier\s+d['']arts\s+graphiques\b[^\n]{0,120}",
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

    text = re.sub(r"['ʼ`´]", "'", text)
    text = re.sub(r"[–—−]", "-", text)

    text = re.sub(r"[^a-zàâäéèêëîïôöùûüçœæ\s'-]", " ", text)
    text = re.sub(r"\b(?![ldjtmnsqc])\w\b", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def light_clean(text: str) -> str:
    text = remove_recurrent_noise(text)
    text = remove_generic_campaign_phrases(text)
    text = normalize_text(text)
    return text


def lemmatize_batch(texts: list[str]) -> list[str]:
    results = []
    for doc in nlp.pipe(texts, batch_size=64):
        results.append(
            " ".join(token.lemma_ for token in doc if not token.is_space)
        )
    return results


def remove_stopwords(text: str) -> str:
    if not isinstance(text, str):
        return ""

    tokens = text.split()
    filtered = [
        token for token in tokens
        if token not in STOPWORDS and len(token) > 1
    ]
    return " ".join(filtered)


def main():
    print("Loading corpus...")
    df = pd.read_csv(INPUT_PATH)
    print(f"Documents loaded: {len(df)}")

    if "text" not in df.columns:
        raise ValueError("Column 'text' is missing from the input file.")

    print(f"Stopwords loaded after protection: {len(STOPWORDS)}")

    print("Light cleaning (text_clean)...")
    tqdm.pandas(desc="Cleaning")
    df["text_clean"] = df["text"].progress_apply(light_clean)

    print("Lemmatizing...")
    df["text_lemmatized"] = lemmatize_batch(df["text_clean"].tolist())

    print("Removing stopwords (text_preprocessed)...")
    tqdm.pandas(desc="Stopwords")
    df["text_preprocessed"] = df["text_lemmatized"].progress_apply(remove_stopwords)

    df = df.drop(columns=["text_lemmatized"])

    df["tokens_before"] = df["text"].apply(
        lambda x: len(str(x).split()) if pd.notna(x) else 0
    )
    df["tokens_clean"] = df["text_clean"].apply(
        lambda x: len(x.split()) if isinstance(x, str) else 0
    )
    df["tokens_after"] = df["text_preprocessed"].apply(
        lambda x: len(x.split()) if isinstance(x, str) else 0
    )

    df["preprocess_keep"] = df["tokens_after"] >= 30
    df_final = df[df["preprocess_keep"]].copy()

    with open(INFO_PATH, "w", encoding="utf-8") as f:
        f.write("=== PREPROCESSING INFO ===\n\n")
        f.write(f"Initial documents: {len(df)}\n")
        f.write(f"Documents kept: {len(df_final)}\n")
        f.write(f"Documents removed: {len(df) - len(df_final)}\n")
        f.write(f"Stopwords used: {len(STOPWORDS)}\n\n")

        f.write("=== TOKEN COUNTS ===\n\n")
        f.write(f"Average tokens (text): {df['tokens_before'].mean():.2f}\n")
        f.write(f"Average tokens (text_clean): {df['tokens_clean'].mean():.2f}\n")
        f.write(f"Average tokens (text_preprocessed): {df['tokens_after'].mean():.2f}\n")

        if df["tokens_before"].mean() > 0:
            reduction = 100 * (
                1 - df["tokens_after"].mean() / df["tokens_before"].mean()
            )
            f.write(f"Average reduction (raw -> preprocessed): {reduction:.2f}%\n\n")

        f.write("Token distribution (preprocessed):\n")
        f.write(df["tokens_after"].describe().to_string())
        f.write("\n\n")

        if "year" in df.columns:
            f.write("=== TOKENS BY YEAR (preprocessed) ===\n\n")
            f.write(
                df.groupby("year")["tokens_after"]
                .describe()[["count", "mean", "min", "50%", "max"]]
                .to_string()
            )
            f.write("\n\n")

        f.write("=== POLITICAL WORDS KEPT ===\n\n")
        f.write(", ".join(sorted(POLITICAL_WORDS_TO_KEEP)))
        f.write("\n")

    df_final.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    # Figure: token reduction

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Boxplot before vs after, by year
    if "year" in df_final.columns:
        years = sorted(df_final["year"].dropna().unique())
        before_data = [df_final[df_final["year"] == y]["tokens_clean"].values for y in years]
        after_data = [df_final[df_final["year"] == y]["tokens_after"].values for y in years]

        positions_b = [i - 0.2 for i in range(len(years))]
        positions_a = [i + 0.2 for i in range(len(years))]
        bp1 = axes[0].boxplot(before_data, positions=positions_b, widths=0.35, patch_artist=True, showfliers=False)
        bp2 = axes[0].boxplot(after_data, positions=positions_a, widths=0.35, patch_artist=True, showfliers=False)
        for box in bp1["boxes"]:
            box.set_facecolor("#a6c8ed")
        for box in bp2["boxes"]:
            box.set_facecolor("#4a7ab5")
        axes[0].set_xticks(range(len(years)))
        axes[0].set_xticklabels([str(y) for y in years])
        axes[0].set_xlabel("Year")
        axes[0].set_ylabel("Tokens per document")
        axes[0].set_title("Token count before (light) vs after (lemma+stopwords) per year")
        axes[0].legend([bp1["boxes"][0], bp2["boxes"][0]], ["text_clean", "text_preprocessed"], loc="upper right")

    # Reduction ratio
    df_final["reduction_pct"] = 100 * (1 - df_final["tokens_after"] / df_final["tokens_clean"].replace(0, np.nan))
    axes[1].hist(df_final["reduction_pct"].dropna(), bins=40, color="#4a7ab5", alpha=0.85)
    axes[1].axvline(df_final["reduction_pct"].median(), color="red", linestyle="--",
                    label=f"median = {df_final['reduction_pct'].median():.1f}%")
    axes[1].set_xlabel("Token reduction (%)")
    axes[1].set_ylabel("Number of documents")
    axes[1].set_title("Token reduction after preprocessing")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(FIG_DIR / "token_reduction.png", dpi=150)
    plt.close()

    print("Done.")
    print(f"Saved corpus: {OUTPUT_PATH}")
    print(f"Saved info: {INFO_PATH}")
    print(f"Saved figure: {FIG_DIR / 'token_reduction.png'}")


if __name__ == "__main__":
    main()
