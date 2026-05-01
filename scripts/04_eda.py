"""
Step 4 — Family-level EDA on text_clean: top words and keyword trends per
political family.
"""

import re
import sys
import unicodedata
from collections import Counter
from itertools import chain
from pathlib import Path

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
TOP_WORDS_FAMILY_PATH = STEP_DIR / "eda_top_words_by_family.csv"
KEYWORD_TRENDS_FAMILY_PATH = STEP_DIR / "eda_keyword_trends_by_family.csv"
KEYWORD_TRENDS_FAMILY_FIG = FIG_DIR / "eda_keyword_trends_by_family.png"
TOP_WORDS_FAMILY_FIG = FIG_DIR / "eda_top_words_by_family.png"


TEXT_COL = "text_clean"
YEAR_COL = "year"

TOPN = 20
MIN_WORD_LEN = 3
KEYWORDS = ["travail", "emploi", "republique", "immigration", "securite", "europe"]


def strip_accents(text):
    if not isinstance(text, str):
        return ""
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


def load_stopwords(path):
    words = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            word = line.strip().lower()
            if word:
                words.add(strip_accents(word))
    return words


def tokenize(text, stopwords):
    if not isinstance(text, str) or not text.strip():
        return []
    text = strip_accents(text.lower())
    tokens = re.findall(r"[a-z]+", text)
    return [t for t in tokens if len(t) >= MIN_WORD_LEN and t not in stopwords]


def main():
    print("Loading corpus...")
    df = pd.read_csv(INPUT_PATH)
    print(f"Documents loaded: {len(df)}")

    stopwords = load_stopwords(STOPWORDS_PATH)

    print("Tokenizing...")
    df["tokens"] = df[TEXT_COL].apply(lambda x: tokenize(x, stopwords))

    print("Computing party_family mapping...")
    df["party_family"] = df.apply(assign_party_family, axis=1)
    df_family = df[~df["party_family"].isin(["unclassified", "other"])].copy()

    print("Computing top words by family...")
    rows = []
    for fam, sub in df_family.groupby("party_family"):
        tokens = list(chain.from_iterable(sub["tokens"]))
        if not tokens:
            continue
        counts = Counter(tokens)
        total = sum(counts.values())
        for word, count in counts.most_common(TOPN):
            rows.append({"party_family": fam, "word": word, "count": count, "share": count / total})
    top_family = pd.DataFrame(rows)
    top_family.to_csv(TOP_WORDS_FAMILY_PATH, index=False, encoding="utf-8-sig")

    families = sorted(df_family["party_family"].unique())
    cols = 4
    rows_grid = (len(families) + cols - 1) // cols
    fig, axes = plt.subplots(rows_grid, cols, figsize=(4 * cols, 2.4 * rows_grid), squeeze=False)
    for idx, fam in enumerate(families):
        sub = top_family[top_family["party_family"] == fam].head(10)
        ax = axes[idx // cols][idx % cols]
        ax.barh(sub["word"][::-1], sub["share"][::-1], color="#4a7ab5")
        ax.set_title(fam, fontsize=10)
        ax.tick_params(axis="both", labelsize=8)
    for idx in range(len(families), rows_grid * cols):
        axes[idx // cols][idx % cols].axis("off")
    plt.suptitle("Top 10 words per political family (share of tokens)", y=1.0)
    plt.tight_layout()
    plt.savefig(TOP_WORDS_FAMILY_FIG, dpi=150, bbox_inches="tight")
    plt.close()

    print("Computing keyword trends by family x year...")
    keywords_norm = [strip_accents(k.lower()) for k in KEYWORDS]
    rows = []
    for (year, fam), sub in df_family.groupby([YEAR_COL, "party_family"]):
        tokens = list(chain.from_iterable(sub["tokens"]))
        total = len(tokens)
        counts = Counter(tokens)
        for kw in keywords_norm:
            rows.append({
                YEAR_COL: year,
                "party_family": fam,
                "keyword": kw,
                "count": counts.get(kw, 0),
                "share": counts.get(kw, 0) / total if total else 0.0,
            })
    family_keyword = pd.DataFrame(rows)
    family_keyword.to_csv(KEYWORD_TRENDS_FAMILY_PATH, index=False, encoding="utf-8-sig")

    n_kw = len(keywords_norm)
    cols = 3
    rows_grid = (n_kw + cols - 1) // cols
    fig, axes = plt.subplots(rows_grid, cols, figsize=(5 * cols, 3 * rows_grid), squeeze=False)
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
    for idx in range(n_kw, rows_grid * cols):
        axes[idx // cols][idx % cols].axis("off")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(7, len(handles)),
               fontsize=8, bbox_to_anchor=(0.5, -0.02))
    plt.suptitle("Keyword share by family across years", y=1.0)
    plt.tight_layout()
    plt.savefig(KEYWORD_TRENDS_FAMILY_FIG, dpi=150, bbox_inches="tight")
    plt.close()

    family_counts = df["party_family"].value_counts()
    with open(INFO_PATH, "w", encoding="utf-8") as f:
        f.write("=== EDA SUMMARY (family-level) ===\n\n")
        f.write(f"Documents: {len(df)}\n")
        f.write(f"Top N words: {TOPN}\n")
        f.write(f"Keywords: {', '.join(KEYWORDS)}\n\n")
        f.write("=== Family-level distribution ===\n")
        f.write(family_counts.to_string())
        f.write("\n")

    print("Done.")


if __name__ == "__main__":
    main()
