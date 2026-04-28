"""
Step 7 — Document Topic Vectors + 2D Projection

Inputs:
data/chunks_with_topics.csv
data/chunks_with_topics_lda.csv

Outputs:
outputs/doc_topic_vectors_bertopic.csv
outputs/doc_topic_vectors_lda.csv
outputs/doc_topic_proj_bertopic.csv
outputs/doc_topic_proj_lda.csv
outputs/doc_topic_projection_info.txt
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List
import argparse

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


# =====================
# Paths
# =====================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

BERTOPIC_CHUNKS_PATH = DATA_DIR / "chunks_with_topics.csv"
LDA_CHUNKS_PATH = DATA_DIR / "chunks_with_topics_lda.csv"

BERTOPIC_VEC_PATH = OUTPUT_DIR / "doc_topic_vectors_bertopic.csv"
LDA_VEC_PATH = OUTPUT_DIR / "doc_topic_vectors_lda.csv"

BERTOPIC_PROJ_PATH = OUTPUT_DIR / "doc_topic_proj_bertopic.csv"
LDA_PROJ_PATH = OUTPUT_DIR / "doc_topic_proj_lda.csv"

INFO_PATH = OUTPUT_DIR / "doc_topic_projection_info.txt"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =====================
# Parameters
# =====================

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

DEFAULT_METHOD = "pca"  # or tsne
DEFAULT_RANDOM_STATE = 42
DEFAULT_TSNE_PERPLEXITY = 30
DEFAULT_TSNE_LEARNING_RATE = 200


# =====================
# Helpers
# =====================

def _safe_perplexity(n_samples: int, desired: int) -> int:
    if n_samples < 3:
        return 2
    return max(2, min(desired, (n_samples - 1) // 3))


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
    pivot.columns = [f"topic_{c}" for c in pivot.columns]

    return pivot


def project_vectors(
    vectors: pd.DataFrame,
    method: str,
    random_state: int,
    tsne_perplexity: int,
    tsne_learning_rate: int,
) -> np.ndarray:
    if method == "pca":
        return PCA(n_components=2, random_state=random_state).fit_transform(vectors.values)

    if method == "tsne":
        perplexity = _safe_perplexity(len(vectors), tsne_perplexity)
        return TSNE(
            n_components=2,
            perplexity=perplexity,
            learning_rate=tsne_learning_rate,
            init="pca",
            random_state=random_state,
        ).fit_transform(vectors.values)

    raise ValueError("Unknown method. Use 'pca' or 'tsne'.")


def doc_metadata(df: pd.DataFrame, doc_col: str, meta_cols: List[str]) -> pd.DataFrame:
    cols = [c for c in meta_cols if c in df.columns]
    if not cols:
        return pd.DataFrame({doc_col: df[doc_col].unique()})

    meta = df[[doc_col] + cols].groupby(doc_col, as_index=False).first()
    return meta


# =====================
# Main
# =====================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build document topic vectors and 2D projections (PCA/TSNE)."
    )
    parser.add_argument("--method", type=str, default=DEFAULT_METHOD, choices=["pca", "tsne"])
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument("--tsne-perplexity", type=int, default=DEFAULT_TSNE_PERPLEXITY)
    parser.add_argument("--tsne-learning-rate", type=int, default=DEFAULT_TSNE_LEARNING_RATE)
    parser.add_argument("--include-outliers", action="store_true")
    parser.add_argument("--lda-use-score", action="store_true")

    args = parser.parse_args()

    print("\nLoading BERTopic chunks...")
    df_bertopic = pd.read_csv(BERTOPIC_CHUNKS_PATH)
    print(f"Chunks loaded: {len(df_bertopic)}")

    if DOC_ID_COL not in df_bertopic.columns or BERTOPIC_TOPIC_COL not in df_bertopic.columns:
        raise ValueError("Missing columns in BERTopic chunks output.")

    print("\nLoading LDA chunks...")
    df_lda = pd.read_csv(LDA_CHUNKS_PATH)
    print(f"Chunks loaded: {len(df_lda)}")

    if DOC_ID_COL not in df_lda.columns or LDA_TOPIC_COL not in df_lda.columns:
        raise ValueError("Missing columns in LDA chunks output.")

    # =====================
    # BERTopic vectors
    # =====================

    drop_topics = [] if args.include_outliers else [-1]
    bertopic_vectors = build_doc_topic_matrix(
        df_bertopic,
        doc_col=DOC_ID_COL,
        topic_col=BERTOPIC_TOPIC_COL,
        weight_col=None,
        drop_topics=drop_topics,
    )

    bertopic_meta = doc_metadata(df_bertopic, DOC_ID_COL, DOC_META_COLS)
    bertopic_vectors_out = bertopic_meta.merge(
        bertopic_vectors.reset_index(),
        on=DOC_ID_COL,
        how="left",
    ).fillna(0.0)

    bertopic_vectors_out.to_csv(BERTOPIC_VEC_PATH, index=False, encoding="utf-8-sig")

    # =====================
    # LDA vectors
    # =====================

    weight_col = LDA_SCORE_COL if args.lda_use_score and LDA_SCORE_COL in df_lda.columns else None
    lda_vectors = build_doc_topic_matrix(
        df_lda,
        doc_col=DOC_ID_COL,
        topic_col=LDA_TOPIC_COL,
        weight_col=weight_col,
        drop_topics=None,
    )

    lda_meta = doc_metadata(df_lda, DOC_ID_COL, DOC_META_COLS)
    lda_vectors_out = lda_meta.merge(
        lda_vectors.reset_index(),
        on=DOC_ID_COL,
        how="left",
    ).fillna(0.0)

    lda_vectors_out.to_csv(LDA_VEC_PATH, index=False, encoding="utf-8-sig")

    # =====================
    # Projection
    # =====================

    print("\nProjecting BERTopic vectors...")
    bertopic_proj = project_vectors(
        bertopic_vectors,
        method=args.method,
        random_state=args.random_state,
        tsne_perplexity=args.tsne_perplexity,
        tsne_learning_rate=args.tsne_learning_rate,
    )

    bertopic_proj_df = bertopic_meta.copy()
    bertopic_proj_df["x"] = bertopic_proj[:, 0]
    bertopic_proj_df["y"] = bertopic_proj[:, 1]
    bertopic_proj_df.to_csv(BERTOPIC_PROJ_PATH, index=False, encoding="utf-8-sig")

    print("\nProjecting LDA vectors...")
    lda_proj = project_vectors(
        lda_vectors,
        method=args.method,
        random_state=args.random_state,
        tsne_perplexity=args.tsne_perplexity,
        tsne_learning_rate=args.tsne_learning_rate,
    )

    lda_proj_df = lda_meta.copy()
    lda_proj_df["x"] = lda_proj[:, 0]
    lda_proj_df["y"] = lda_proj[:, 1]
    lda_proj_df.to_csv(LDA_PROJ_PATH, index=False, encoding="utf-8-sig")

    # =====================
    # Summary
    # =====================

    with open(INFO_PATH, "w", encoding="utf-8") as f:
        f.write("=== Document Topic Projection Info ===\n\n")
        f.write(f"Method: {args.method}\n")
        f.write(f"Random state: {args.random_state}\n")
        f.write(f"TSNE perplexity: {args.tsne_perplexity}\n")
        f.write(f"TSNE learning rate: {args.tsne_learning_rate}\n")
        f.write(f"Include BERTopic outliers: {args.include_outliers}\n")
        f.write(f"LDA use score: {args.lda_use_score}\n\n")

        f.write("=== BERTopic ===\n")
        f.write(f"Docs: {len(bertopic_vectors)}\n")
        f.write(f"Topics (columns): {bertopic_vectors.shape[1]}\n\n")

        f.write("=== LDA ===\n")
        f.write(f"Docs: {len(lda_vectors)}\n")
        f.write(f"Topics (columns): {lda_vectors.shape[1]}\n")

    print("\nDone.")
    print(f"Saved BERTopic vectors: {BERTOPIC_VEC_PATH}")
    print(f"Saved LDA vectors: {LDA_VEC_PATH}")
    print(f"Saved BERTopic projection: {BERTOPIC_PROJ_PATH}")
    print(f"Saved LDA projection: {LDA_PROJ_PATH}")
    print(f"Saved summary: {INFO_PATH}")


if __name__ == "__main__":
    main()
