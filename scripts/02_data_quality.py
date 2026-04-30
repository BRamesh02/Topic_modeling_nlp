"""
Step 2 — OCR quality scoring, filter, and lexical diagnostics on the joined
corpus.
"""

from pathlib import Path
from collections import Counter
import re

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


# Used only for the diagnostic plots, not for filtering.
DIAGNOSTIC_STOPWORDS = set("""
le la les de des du un une et en à au aux ce ces son ses sa se ne pas que qui
il elle on nous vous ils elles je tu est sont par pour sur avec dans plus mais
ou où donc car ni si tout tous toute toutes très bien aussi comme même sans
sous entre vers chez cette cet dont peut après avant encore autre autres peu
puis été faire avoir fait dit lors non oui quand quel quelle quels quelles
ont sera faut doit déjà depuis ainsi jusque jusqu chaque ceux celles celui celle
alors notre votre leur leurs mes tes mon ma ton ta lui eux moi toi ici là
""".split())


def extract_tokens(text: str) -> list[str]:
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
        "score": score
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
    df.boxplot(column="tokens", by="year", grid=False, showfliers=False)
    plt.title("Document length by year")
    plt.suptitle("")
    plt.xlabel("Year")
    plt.ylabel("Tokens")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "text_length_by_year.png", dpi=150)
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


def make_lexical_diagnostic_plots(df):
    all_tokens = []

    for text in df["text"].dropna():
        all_tokens.extend(extract_tokens(text))

    token_counts = Counter(all_tokens)
    total_tokens = sum(token_counts.values())

    if total_tokens == 0:
        print("No tokens found for lexical diagnostics.")
        return

    # 1. Top most frequent words
    top_words = token_counts.most_common(30)
    words = [w for w, _ in top_words]
    counts = [c for _, c in top_words]

    plt.figure(figsize=(10, 7))
    plt.barh(words[::-1], counts[::-1])
    plt.title("Most frequent words before stopword removal")
    plt.xlabel("Frequency")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "top_words_before_stopwords.png", dpi=150)
    plt.close()

    # 2. Top stopwords only
    stopword_counts = {
        w: c for w, c in token_counts.items()
        if w in DIAGNOSTIC_STOPWORDS
    }

    top_stopwords = Counter(stopword_counts).most_common(30)

    if top_stopwords:
        words = [w for w, _ in top_stopwords]
        counts = [c for _, c in top_stopwords]

        plt.figure(figsize=(10, 7))
        plt.barh(words[::-1], counts[::-1])
        plt.title("Most frequent stopwords before preprocessing")
        plt.xlabel("Frequency")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "top_stopwords_before_preprocessing.png", dpi=150)
        plt.close()

    # 3. Share of stopwords by year
    rows = []

    for year, sub in df.groupby("year"):
        year_tokens = []
        for text in sub["text"].dropna():
            year_tokens.extend(extract_tokens(text))

        if len(year_tokens) == 0:
            continue

        stop_count = sum(t in DIAGNOSTIC_STOPWORDS for t in year_tokens)
        rows.append({
            "year": year,
            "stopword_share": stop_count / len(year_tokens),
            "n_tokens": len(year_tokens)
        })

    stop_df = pd.DataFrame(rows)

    if len(stop_df) > 0:
        plt.figure(figsize=(8, 5))
        plt.bar(stop_df["year"].astype(str), stop_df["stopword_share"])
        plt.title("Share of stopwords by year")
        plt.xlabel("Year")
        plt.ylabel("Stopword share")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "stopword_share_by_year.png", dpi=150)
        plt.close()

    # 4. Cumulative frequency of top words
    sorted_counts = [c for _, c in token_counts.most_common(100)]
    cumulative = pd.Series(sorted_counts).cumsum() / total_tokens

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(cumulative) + 1), cumulative)
    plt.title("Cumulative share of top 100 words")
    plt.xlabel("Top N words")
    plt.ylabel("Cumulative share of all tokens")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "top_words_cumulative_share.png", dpi=150)
    plt.close()

    # Save lexical stats
    lexical_stats_path = REPORTS_DIR / "lexical_diagnostics.txt"
    with open(lexical_stats_path, "w", encoding="utf-8") as f:
        f.write("=== LEXICAL DIAGNOSTICS BEFORE STOPWORD REMOVAL ===\n\n")
        f.write(f"Total tokens: {total_tokens}\n")
        f.write(f"Unique tokens: {len(token_counts)}\n\n")

        f.write("Top 50 words:\n")
        for word, count in token_counts.most_common(50):
            f.write(f"{word}: {count} ({count / total_tokens:.4%})\n")

        f.write("\nTop 50 diagnostic stopwords:\n")
        for word, count in Counter(stopword_counts).most_common(50):
            f.write(f"{word}: {count} ({count / total_tokens:.4%})\n")

        if len(stop_df) > 0:
            f.write("\nStopword share by year:\n")
            f.write(stop_df.to_string(index=False))
            f.write("\n")

    print(f"Saved lexical diagnostics → {lexical_stats_path}")


def main():
    print("Loading dataset...")
    df = pd.read_csv(INPUT_PATH)

    if "text" not in df.columns:
        raise ValueError("Missing column: text")

    print("Computing OCR quality...")
    metrics = df["text"].apply(evaluate_text).apply(pd.Series)
    df = pd.concat([df, metrics], axis=1)

    df["keep"] = (
        (df["tokens"] > 50) &
        (df["short_ratio"] < 0.5)
    )

    df_clean = df[df["keep"]].copy()

    with open(STATS_PATH, "w", encoding="utf-8") as f:
        f.write("=== OCR QUALITY ===\n\n")
        f.write(f"Total documents: {len(df)}\n")
        f.write(f"Kept documents: {len(df_clean)}\n")
        f.write(f"Removed: {len(df) - len(df_clean)}\n\n")

        f.write("Average score:\n")
        f.write(str(df.groupby("year")["score"].mean()))
        f.write("\n\n")

        f.write("Median length:\n")
        f.write(str(df.groupby("year")["tokens"].median()))
        f.write("\n")

    print("Generating OCR quality plots...")
    make_quality_plots(df)

    print("Generating lexical diagnostic plots...")
    make_lexical_diagnostic_plots(df_clean)

    df_clean.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print("Done.")
    print(f"Saved dataset → {OUTPUT_PATH}")
    print(f"Saved stats → {STATS_PATH}")
    print(f"Saved figures → {FIG_DIR}")


if __name__ == "__main__":
    main()