"""
Step 9 — Aggregate chunk-level topics into document-level topic vectors and
attach a party_family label per document. Used by steps 10, 11 and 12.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, List
import argparse

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _family_mapping import AFFILIATION_FIELDS, assign_party_family


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = PROJECT_ROOT / "outputs"

BERTOPIC_DIR = OUTPUTS / "07_bertopic"
LDA_DIR = OUTPUTS / "08_lda"
STEP_DIR = OUTPUTS / "09_doc_topic_vectors"
REPORTS_DIR = STEP_DIR / "reports"
FIG_DIR = STEP_DIR / "figures"

STEP_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

BERTOPIC_CHUNKS_PATH = BERTOPIC_DIR / "chunks_with_topics.csv"
LDA_CHUNKS_PATH = LDA_DIR / "chunks_with_topics_lda.csv"

BERTOPIC_VEC_PATH = STEP_DIR / "doc_topic_vectors_bertopic.csv"
LDA_VEC_PATH = STEP_DIR / "doc_topic_vectors_lda.csv"
PARTY_FAMILY_PATH = STEP_DIR / "doc_party_family.csv"
INFO_PATH = REPORTS_DIR / "doc_topic_vectors_info.txt"


DOC_ID_COL = "doc_id"
BERTOPIC_TOPIC_COL = "topic"
LDA_TOPIC_COL = "topic_lda"
LDA_SCORE_COL = "topic_lda_score"

DOC_META_COLS = [
    "year",
    "date",
    "contexte-tour",
    "titulaire-liste",
    "titulaire-soutien",
    "titulaire-sexe",
    "titulaire-prenom",
    "titulaire-nom",
]

def build_doc_topic_matrix(
    df: pd.DataFrame,
    doc_col: str,
    topic_col: str,
    weight_col: str | None = None,
    drop_topics: Iterable[int] | None = None,
) -> pd.DataFrame:
    work = df[[doc_col, topic_col]].copy()
    if weight_col and weight_col in df.columns:
        work["weight"] = df[weight_col].astype(float)
    else:
        work["weight"] = 1.0

    if drop_topics is not None:
        work = work[~work[topic_col].isin(drop_topics)]

    pivot = (
        work.pivot_table(
            index=doc_col,
            columns=topic_col,
            values="weight",
            aggfunc="sum",
            fill_value=0.0,
        )
        .sort_index(axis=1)
    )

    row_sums = pivot.sum(axis=1)
    pivot = pivot.div(row_sums.replace(0, np.nan), axis=0).fillna(0.0)
    pivot.columns = [f"topic_{int(c)}" for c in pivot.columns]
    return pivot


def doc_metadata(df: pd.DataFrame, doc_col: str, meta_cols: List[str]) -> pd.DataFrame:
    cols = [c for c in meta_cols if c in df.columns]
    if not cols:
        return pd.DataFrame({doc_col: df[doc_col].unique()})
    return df[[doc_col] + cols].groupby(doc_col, as_index=False).first()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build doc-level topic vectors and assign party families."
    )
    parser.add_argument("--include-bertopic-outliers", action="store_true",
                        help="Keep topic -1 from BERTopic (default: drop).")
    parser.add_argument("--lda-use-score", action="store_true",
                        help="Weight LDA chunk-topic counts by topic_lda_score.")

    args = parser.parse_args()

    print("\nLoading BERTopic chunks...")
    df_bert = pd.read_csv(BERTOPIC_CHUNKS_PATH)
    print(f"Chunks loaded: {len(df_bert)}")

    print("\nLoading LDA chunks...")
    df_lda = pd.read_csv(LDA_CHUNKS_PATH)
    print(f"Chunks loaded: {len(df_lda)}")

    if DOC_ID_COL not in df_bert.columns or BERTOPIC_TOPIC_COL not in df_bert.columns:
        raise ValueError("Missing columns in BERTopic chunks output.")
    if DOC_ID_COL not in df_lda.columns or LDA_TOPIC_COL not in df_lda.columns:
        raise ValueError("Missing columns in LDA chunks output.")


    drop_topics = [] if args.include_bertopic_outliers else [-1]
    bert_vec = build_doc_topic_matrix(
        df_bert,
        doc_col=DOC_ID_COL,
        topic_col=BERTOPIC_TOPIC_COL,
        weight_col=None,
        drop_topics=drop_topics,
    )

    bert_meta = doc_metadata(df_bert, DOC_ID_COL, DOC_META_COLS)
    bert_meta["party_family"] = bert_meta.apply(assign_party_family, axis=1)

    bert_out = bert_meta.merge(
        bert_vec.reset_index(),
        on=DOC_ID_COL,
        how="left",
    ).fillna(0.0)

    topic_cols_bert = [c for c in bert_out.columns if c.startswith("topic_")]
    if topic_cols_bert:
        bert_out["dominant_topic"] = (
            bert_out[topic_cols_bert].idxmax(axis=1).str.replace("topic_", "").astype(int)
        )

    bert_out.to_csv(BERTOPIC_VEC_PATH, index=False, encoding="utf-8-sig")


    weight_col = LDA_SCORE_COL if args.lda_use_score and LDA_SCORE_COL in df_lda.columns else None
    lda_vec = build_doc_topic_matrix(
        df_lda,
        doc_col=DOC_ID_COL,
        topic_col=LDA_TOPIC_COL,
        weight_col=weight_col,
        drop_topics=None,
    )

    lda_meta = doc_metadata(df_lda, DOC_ID_COL, DOC_META_COLS)
    lda_meta["party_family"] = lda_meta.apply(assign_party_family, axis=1)

    lda_out = lda_meta.merge(
        lda_vec.reset_index(),
        on=DOC_ID_COL,
        how="left",
    ).fillna(0.0)

    topic_cols_lda = [c for c in lda_out.columns if c.startswith("topic_")]
    if topic_cols_lda:
        lda_out["dominant_topic"] = (
            lda_out[topic_cols_lda].idxmax(axis=1).str.replace("topic_", "").astype(int)
        )

    lda_out.to_csv(LDA_VEC_PATH, index=False, encoding="utf-8-sig")


    family = (
        bert_meta[[DOC_ID_COL, "party_family"] + [c for c in DOC_META_COLS if c in bert_meta.columns]]
        .copy()
    )
    family.to_csv(PARTY_FAMILY_PATH, index=False, encoding="utf-8-sig")


    with open(INFO_PATH, "w", encoding="utf-8") as f:
        f.write("=== Doc-level Topic Vectors ===\n\n")
        f.write(f"BERTopic docs: {len(bert_out)}, topics: {len(topic_cols_bert)}\n")
        f.write(f"LDA docs: {len(lda_out)}, topics: {len(topic_cols_lda)}\n")
        f.write(f"BERTopic outliers dropped: {not args.include_bertopic_outliers}\n")
        f.write(f"LDA weighted by score: {args.lda_use_score}\n\n")

        f.write("=== Party family counts (BERTopic docs) ===\n")
        f.write(bert_out["party_family"].value_counts().to_string())
        f.write("\n\n")

        if "year" in bert_out.columns:
            f.write("=== Docs by year ===\n")
            f.write(bert_out["year"].value_counts().sort_index().to_string())
            f.write("\n\n")

        if "dominant_topic" in bert_out.columns:
            f.write("=== Dominant BERTopic topics ===\n")
            f.write(bert_out["dominant_topic"].value_counts().head(20).to_string())
            f.write("\n")


    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Party family counts
    counts = bert_out["party_family"].value_counts()
    colors = ["#4a7ab5"] * len(counts)
    counts.plot(kind="barh", ax=axes[0], color=colors)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Number of documents")
    axes[0].set_title("Documents per party family")
    for i, v in enumerate(counts.values):
        axes[0].text(v + max(counts.values) * 0.01, i, str(v), va="center", fontsize=9)

    # Crosstab year x family
    if "year" in bert_out.columns:
        ct = pd.crosstab(bert_out["year"], bert_out["party_family"])
        valid = [c for c in ct.columns if c not in ("unclassified", "other")]
        ct_valid = ct[valid] if valid else ct
        ct_valid.plot(kind="bar", stacked=True, ax=axes[1], colormap="tab10")
        axes[1].set_xlabel("Year")
        axes[1].set_ylabel("Number of documents")
        axes[1].set_title("Party family distribution by year")
        axes[1].legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)
        axes[1].tick_params(axis="x", rotation=0)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "party_family_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("\nDone.")
    print(f"Saved BERTopic doc vectors: {BERTOPIC_VEC_PATH}")
    print(f"Saved LDA doc vectors: {LDA_VEC_PATH}")
    print(f"Saved party_family table: {PARTY_FAMILY_PATH}")
    print(f"Saved figures: {FIG_DIR}")
    print(f"Saved summary: {INFO_PATH}")


if __name__ == "__main__":
    main()
