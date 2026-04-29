"""
Step 4 — Exploratory data analysis on the cleaned text (text_clean column of
corpus_preprocessed.csv): top words by year and by party, keyword trends, and
family-level distributions (using the shared family mapping).

Runs after step 3 (preprocessing) so that institutional OCR boilerplate and
generic campaign phrases have already been stripped.
"""

from __future__ import annotations

import sys
from pathlib import Path
from collections import Counter
from itertools import chain
from typing import Iterable, List
import argparse
import re
import unicodedata

import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = PROJECT_ROOT / "outputs"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _family_mapping import assign_party_family

PREV_DIR = OUTPUTS / "03_preprocessing"
STEP_DIR = OUTPUTS / "04_eda"
FIG_DIR = STEP_DIR / "figures"
REPORTS_DIR = STEP_DIR / "reports"

STEP_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

INPUT_PATH = PREV_DIR / "corpus_preprocessed.csv"
STOPWORDS_PATH = PROJECT_ROOT / "stop_word_fr.txt"

INFO_PATH = REPORTS_DIR / "eda_info.txt"
TOP_WORDS_YEAR_PATH = STEP_DIR / "eda_top_words_by_year.csv"
TOP_WORDS_PARTY_PATH = STEP_DIR / "eda_top_words_by_party.csv"
TOP_WORDS_FAMILY_PATH = STEP_DIR / "eda_top_words_by_family.csv"
KEYWORD_TRENDS_PATH = STEP_DIR / "eda_keyword_trends.csv"
KEYWORD_TRENDS_FAMILY_PATH = STEP_DIR / "eda_keyword_trends_by_family.csv"
KEYWORD_TRENDS_FIG = FIG_DIR / "eda_keyword_trends.png"
KEYWORD_TRENDS_FAMILY_FIG = FIG_DIR / "eda_keyword_trends_by_family.png"
TOP_WORDS_FAMILY_FIG = FIG_DIR / "eda_top_words_by_family.png"


TEXT_COL = "text_clean"
YEAR_COL = "year"
DOC_ID_COL = "id"

DEFAULT_PARTY_COL = "titulaire-soutien"
DEFAULT_TOPN = 20
DEFAULT_MIN_WORD_LEN = 3
DEFAULT_MIN_DOCS_PER_PARTY = 10
DEFAULT_KEYWORDS = [
    "travail",
    "emploi",
    "republique",
    "immigration",
    "securite",
    "europe",
]


def strip_accents(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return (
        unicodedata.normalize("NFKD", text)
        .encode("ascii", "ignore")
        .decode("ascii")
    )


def load_stopwords(path: Path) -> set[str]:
    if not path.exists():
        return set()

    words = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            word = line.strip().lower()
            if word:
                words.add(strip_accents(word))

    return words


def tokenize(text: str, min_len: int, stopwords: set[str]) -> List[str]:
    if not isinstance(text, str) or not text.strip():
        return []

    text = strip_accents(text.lower())
    tokens = re.findall(r"[a-z]+", text)
    tokens = [t for t in tokens if len(t) >= min_len]
    if stopwords:
        tokens = [t for t in tokens if t not in stopwords]
    return tokens


def top_words_by_group(
    df: pd.DataFrame,
    group_col: str,
    topn: int,
) -> pd.DataFrame:
    rows = []

    for group_value, sub in df.groupby(group_col):
        tokens = list(chain.from_iterable(sub["tokens"]))
        if not tokens:
            continue
        counts = Counter(tokens)
        total = sum(counts.values())
        for word, count in counts.most_common(topn):
            rows.append(
                {
                    group_col: group_value,
                    "word": word,
                    "count": count,
                    "share": count / total,
                }
            )

    return pd.DataFrame(rows)


def keyword_trends_by_year(
    df: pd.DataFrame,
    year_col: str,
    keywords: Iterable[str],
) -> pd.DataFrame:
    keywords = [strip_accents(k.lower()) for k in keywords]
    rows = []

    for year_value, sub in df.groupby(year_col):
        tokens = list(chain.from_iterable(sub["tokens"]))
        total = len(tokens)
        counts = Counter(tokens)
        for kw in keywords:
            rows.append(
                {
                    year_col: year_value,
                    "keyword": kw,
                    "count": counts.get(kw, 0),
                    "share": counts.get(kw, 0) / total if total else 0.0,
                }
            )

    return pd.DataFrame(rows)


def plot_keyword_trends(df: pd.DataFrame, year_col: str, output_path: Path) -> None:
    if df.empty:
        return

    plt.figure(figsize=(9, 5))
    for keyword, sub in df.groupby("keyword"):
        sub = sub.sort_values(year_col)
        plt.plot(sub[year_col].astype(str), sub["share"], marker="o", label=keyword)

    plt.title("Keyword share by year")
    plt.xlabel("Year")
    plt.ylabel("Share of tokens")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="EDA: word usage by year/party.")
    parser.add_argument("--party-col", type=str, default=DEFAULT_PARTY_COL)
    parser.add_argument("--topn", type=int, default=DEFAULT_TOPN)
    parser.add_argument("--min-word-len", type=int, default=DEFAULT_MIN_WORD_LEN)
    parser.add_argument("--min-docs-per-party", type=int, default=DEFAULT_MIN_DOCS_PER_PARTY)
    parser.add_argument("--keywords", type=str, nargs="+", default=DEFAULT_KEYWORDS)
    parser.add_argument("--no-stopwords", action="store_true")

    args = parser.parse_args()

    print("Loading corpus...")
    df = pd.read_csv(INPUT_PATH)
    print(f"Documents loaded: {len(df)}")

    if TEXT_COL not in df.columns:
        raise ValueError(f"Missing column: {TEXT_COL}")

    if YEAR_COL not in df.columns:
        raise ValueError(f"Missing column: {YEAR_COL}")

    if args.party_col not in df.columns:
        raise ValueError(f"Party column not found: {args.party_col}")

    stopwords = set() if args.no_stopwords else load_stopwords(STOPWORDS_PATH)

    print("Tokenizing...")
    df["tokens"] = df[TEXT_COL].apply(
        lambda x: tokenize(x, min_len=args.min_word_len, stopwords=stopwords)
    )

    # Top words by year
    print("Computing top words by year...")
    top_year = top_words_by_group(df, YEAR_COL, args.topn)
    top_year.to_csv(TOP_WORDS_YEAR_PATH, index=False, encoding="utf-8-sig")

    # Top words by party
    print("Computing top words by party...")
    party_counts = df[args.party_col].value_counts(dropna=False)
    allowed_parties = party_counts[party_counts >= args.min_docs_per_party].index

    df_party = df[df[args.party_col].isin(allowed_parties)].copy()
    top_party = top_words_by_group(df_party, args.party_col, args.topn)
    top_party.to_csv(TOP_WORDS_PARTY_PATH, index=False, encoding="utf-8-sig")

    # Keyword trends
    print("Computing keyword trends...")
    trends = keyword_trends_by_year(df, YEAR_COL, args.keywords)
    trends.to_csv(KEYWORD_TRENDS_PATH, index=False, encoding="utf-8-sig")
    plot_keyword_trends(trends, YEAR_COL, KEYWORD_TRENDS_FIG)

    print("Computing party_family mapping...")
    df["party_family"] = df.apply(assign_party_family, axis=1)
    df_family = df[~df["party_family"].isin(["unclassified", "other"])].copy()

    print("Computing top words by family...")
    top_family = top_words_by_group(df_family, "party_family", args.topn)
    top_family.to_csv(TOP_WORDS_FAMILY_PATH, index=False, encoding="utf-8-sig")

    # Family-level top words bar chart (one panel per family)
    families_present = sorted(df_family["party_family"].unique())
    n_fam = len(families_present)
    if n_fam > 0:
        cols = 4
        rows = (n_fam + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 2.4 * rows), squeeze=False)
        for idx, fam in enumerate(families_present):
            sub = top_family[top_family["party_family"] == fam].head(10)
            ax = axes[idx // cols][idx % cols]
            ax.barh(sub["word"][::-1], sub["share"][::-1], color="#4a7ab5")
            ax.set_title(fam, fontsize=10)
            ax.tick_params(axis="both", labelsize=8)
        for idx in range(n_fam, rows * cols):
            axes[idx // cols][idx % cols].axis("off")
        plt.suptitle("Top 10 words per political family (share of tokens)", y=1.0)
        plt.tight_layout()
        plt.savefig(TOP_WORDS_FAMILY_FIG, dpi=150, bbox_inches="tight")
        plt.close()

    # Keyword trends crossed by family x year
    print("Computing keyword trends by family...")
    family_keyword_rows = []
    keywords_norm = [strip_accents(k.lower()) for k in args.keywords]
    for (year, fam), sub in df_family.groupby([YEAR_COL, "party_family"]):
        tokens = list(chain.from_iterable(sub["tokens"]))
        total = len(tokens)
        counts = Counter(tokens)
        for kw in keywords_norm:
            family_keyword_rows.append({
                YEAR_COL: year,
                "party_family": fam,
                "keyword": kw,
                "count": counts.get(kw, 0),
                "share": counts.get(kw, 0) / total if total else 0.0,
            })
    family_keyword = pd.DataFrame(family_keyword_rows)
    family_keyword.to_csv(KEYWORD_TRENDS_FAMILY_PATH, index=False, encoding="utf-8-sig")

    # Plot keyword trends by family (one subplot per keyword)
    if not family_keyword.empty:
        n_kw = len(keywords_norm)
        cols = 3
        rows = (n_kw + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3 * rows), squeeze=False)
        for idx, kw in enumerate(keywords_norm):
            ax = axes[idx // cols][idx % cols]
            sub = family_keyword[family_keyword["keyword"] == kw]
            for fam, g in sub.groupby("party_family"):
                g = g.sort_values(YEAR_COL)
                ax.plot(g[YEAR_COL].astype(str), g["share"], marker="o", label=fam, alpha=0.8)
            ax.set_title(f"keyword: '{kw}'", fontsize=10)
            ax.set_xlabel("Year")
            ax.set_ylabel("Share of tokens")
            ax.grid(alpha=0.3)
        for idx in range(n_kw, rows * cols):
            axes[idx // cols][idx % cols].axis("off")
        handles, labels = axes[0][0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="lower center", ncol=min(7, len(handles)),
                       fontsize=8, bbox_to_anchor=(0.5, -0.02))
        plt.suptitle("Keyword share by family across years", y=1.0)
        plt.tight_layout()
        plt.savefig(KEYWORD_TRENDS_FAMILY_FIG, dpi=150, bbox_inches="tight")
        plt.close()

    # Summary
    family_counts = df["party_family"].value_counts()
    with open(INFO_PATH, "w", encoding="utf-8") as f:
        f.write("=== EDA SUMMARY ===\n\n")
        f.write(f"Documents: {len(df)}\n")
        f.write(f"Year column: {YEAR_COL}\n")
        f.write(f"Party column (raw): {args.party_col}\n")
        f.write(f"Min word length: {args.min_word_len}\n")
        f.write(f"Top N words: {args.topn}\n")
        f.write(f"Stopwords used: {not args.no_stopwords}\n")
        f.write(f"Min docs per party: {args.min_docs_per_party}\n")
        f.write(f"Keywords: {', '.join(args.keywords)}\n\n")

        f.write("=== Family-level distribution (consolidated mapping) ===\n")
        f.write(family_counts.to_string())
        f.write("\n\n")

        f.write("Top words by year saved to: eda_top_words_by_year.csv\n")
        f.write("Top words by raw party saved to: eda_top_words_by_party.csv\n")
        f.write("Top words by political family saved to: eda_top_words_by_family.csv\n")
        f.write("Keyword trends by year saved to: eda_keyword_trends.csv\n")
        f.write("Keyword trends by family saved to: eda_keyword_trends_by_family.csv\n")

    print("Done.")
    print(f"Saved summary: {INFO_PATH}")
    print(f"Saved top words by year: {TOP_WORDS_YEAR_PATH}")
    print(f"Saved top words by party: {TOP_WORDS_PARTY_PATH}")
    print(f"Saved keyword trends: {KEYWORD_TRENDS_PATH}")
    print(f"Saved keyword trends figure: {KEYWORD_TRENDS_FIG}")


if __name__ == "__main__":
    main()
