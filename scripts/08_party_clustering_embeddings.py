"""
Step 8 — Party Clustering from Original Embeddings

Inputs:
data/corpus_chunks.csv
data/chunk_embeddings.npy

Outputs:
outputs/party_clustering.csv
outputs/party_clustering_projection.csv
outputs/party_clustering_info.txt
"""

from __future__ import annotations

from pathlib import Path
from typing import List
import argparse

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score


# =====================
# Paths
# =====================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

CHUNKS_PATH = DATA_DIR / "corpus_chunks.csv"
EMBEDDINGS_PATH = DATA_DIR / "chunk_embeddings.npy"

OUTPUT_CLUSTER_PATH = OUTPUT_DIR / "party_clustering.csv"
OUTPUT_PROJ_PATH = OUTPUT_DIR / "party_clustering_projection.csv"
INFO_PATH = OUTPUT_DIR / "party_clustering_info.txt"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =====================
# Parameters
# =====================

DOC_ID_COL = "doc_id"
CHUNK_WORDS_COL = "chunk_words"

CANDIDATE_PARTY_COLS = [
    "titulaire-soutien",
    "titulaire-liste",
    "parti",
    "party",
]

DEFAULT_METHOD = "pca"  # or tsne
DEFAULT_RANDOM_STATE = 42
DEFAULT_TSNE_PERPLEXITY = 30
DEFAULT_TSNE_LEARNING_RATE = 200
DEFAULT_MIN_DOCS_PER_PARTY = 10
DEFAULT_TOP_PARTIES = 10

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


# =====================
# Helpers
# =====================

def _safe_perplexity(n_samples: int, desired: int) -> int:
    if n_samples < 3:
        return 2
    return max(2, min(desired, (n_samples - 1) // 3))


def resolve_party_col(df: pd.DataFrame, requested: str | None) -> str:
    if requested:
        if requested not in df.columns:
            raise ValueError(f"Party column not found: {requested}")
        return requested

    for col in CANDIDATE_PARTY_COLS:
        if col in df.columns:
            return col

    raise ValueError("No party column found. Use --party-col to specify one.")


def compute_doc_embeddings(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    doc_col: str,
    weight_col: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    doc_ids = df[doc_col].astype(str).values
    unique_docs, inverse = np.unique(doc_ids, return_inverse=True)

    dim = embeddings.shape[1]
    sums = np.zeros((len(unique_docs), dim), dtype=float)
    weights_sum = np.zeros(len(unique_docs), dtype=float)

    if weight_col and weight_col in df.columns:
        weights = df[weight_col].astype(float).values
        np.add.at(sums, inverse, embeddings * weights[:, None])
        np.add.at(weights_sum, inverse, weights)
    else:
        np.add.at(sums, inverse, embeddings)
        np.add.at(weights_sum, inverse, 1.0)

    doc_embeddings = sums / weights_sum[:, None]
    return unique_docs, doc_embeddings


def project_vectors(
    vectors: np.ndarray,
    method: str,
    random_state: int,
    tsne_perplexity: int,
    tsne_learning_rate: int,
) -> np.ndarray:
    if method == "pca":
        return PCA(n_components=2, random_state=random_state).fit_transform(vectors)

    if method == "tsne":
        perplexity = _safe_perplexity(len(vectors), tsne_perplexity)
        return TSNE(
            n_components=2,
            perplexity=perplexity,
            learning_rate=tsne_learning_rate,
            init="pca",
            random_state=random_state,
        ).fit_transform(vectors)

    raise ValueError("Unknown method. Use 'pca' or 'tsne'.")


def cluster_purity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) == 0:
        return float("nan")
    table = pd.crosstab(y_pred, y_true)
    return float(table.max(axis=1).sum() / table.values.sum())


def doc_metadata(df: pd.DataFrame, doc_col: str, meta_cols: List[str]) -> pd.DataFrame:
    cols = [c for c in meta_cols if c in df.columns]
    if not cols:
        return pd.DataFrame({doc_col: df[doc_col].astype(str).unique()})

    meta = df[[doc_col] + cols].groupby(doc_col, as_index=False).first()
    return meta


# =====================
# Main
# =====================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cluster documents by party using original chunk embeddings."
    )
    parser.add_argument("--party-col", type=str, default=None)
    parser.add_argument("--min-docs-per-party", type=int, default=DEFAULT_MIN_DOCS_PER_PARTY)
    parser.add_argument("--top-parties", type=int, default=DEFAULT_TOP_PARTIES)
    parser.add_argument("--n-clusters", type=int, default=None)
    parser.add_argument("--method", type=str, default=DEFAULT_METHOD, choices=["pca", "tsne"])
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument("--tsne-perplexity", type=int, default=DEFAULT_TSNE_PERPLEXITY)
    parser.add_argument("--tsne-learning-rate", type=int, default=DEFAULT_TSNE_LEARNING_RATE)
    parser.add_argument("--weight-by-words", action="store_true")

    args = parser.parse_args()

    print("\nLoading chunks...")
    df = pd.read_csv(CHUNKS_PATH)
    print(f"Chunks loaded: {len(df)}")

    if DOC_ID_COL not in df.columns:
        raise ValueError(f"Missing column: {DOC_ID_COL}")

    print("\nLoading embeddings...")
    embeddings = np.load(EMBEDDINGS_PATH)
    print(f"Embeddings shape: {embeddings.shape}")

    if len(df) != embeddings.shape[0]:
        raise ValueError(
            f"Mismatch: {len(df)} chunks but {embeddings.shape[0]} embeddings."
        )

    party_col = resolve_party_col(df, args.party_col)
    print(f"Using party column: {party_col}")

    print("\nBuilding document embeddings...")
    weight_col = CHUNK_WORDS_COL if args.weight_by_words else None
    doc_ids, doc_embeddings = compute_doc_embeddings(
        df,
        embeddings,
        doc_col=DOC_ID_COL,
        weight_col=weight_col,
    )

    meta = doc_metadata(df, DOC_ID_COL, DOC_META_COLS + [party_col])
    meta[DOC_ID_COL] = meta[DOC_ID_COL].astype(str)

    doc_df = pd.DataFrame({DOC_ID_COL: doc_ids})
    doc_df = doc_df.merge(meta, on=DOC_ID_COL, how="left")

    doc_df[party_col] = doc_df[party_col].fillna("Unknown")

    # Filter parties
    party_counts = doc_df[party_col].value_counts()
    allowed = party_counts[party_counts >= args.min_docs_per_party]
    if args.top_parties and args.top_parties > 0:
        allowed = allowed.head(args.top_parties)

    doc_mask = doc_df[party_col].isin(allowed.index)
    doc_df = doc_df.loc[doc_mask].reset_index(drop=True)
    doc_embeddings = doc_embeddings[doc_mask.values]

    if len(doc_df) == 0:
        raise ValueError("No documents left after filtering parties.")

    n_clusters = args.n_clusters or doc_df[party_col].nunique()

    print(f"\nClustering into {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=args.random_state, n_init="auto")
    clusters = kmeans.fit_predict(doc_embeddings)

    doc_df["cluster"] = clusters

    # Metrics
    y_true = doc_df[party_col].values
    y_pred = clusters

    purity = cluster_purity(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred) if doc_df[party_col].nunique() > 1 else float("nan")
    nmi = normalized_mutual_info_score(y_true, y_pred) if doc_df[party_col].nunique() > 1 else float("nan")

    sil = float("nan")
    if n_clusters > 1 and len(doc_df) > n_clusters:
        sil = float(silhouette_score(doc_embeddings, y_pred))

    # Projection
    print("\nProjecting to 2D...")
    proj = project_vectors(
        doc_embeddings,
        method=args.method,
        random_state=args.random_state,
        tsne_perplexity=args.tsne_perplexity,
        tsne_learning_rate=args.tsne_learning_rate,
    )

    proj_df = doc_df[[DOC_ID_COL, party_col, "cluster"]].copy()
    proj_df["x"] = proj[:, 0]
    proj_df["y"] = proj[:, 1]

    # Save outputs
    doc_df.to_csv(OUTPUT_CLUSTER_PATH, index=False, encoding="utf-8-sig")
    proj_df.to_csv(OUTPUT_PROJ_PATH, index=False, encoding="utf-8-sig")

    with open(INFO_PATH, "w", encoding="utf-8") as f:
        f.write("=== Party Clustering Info ===\n\n")
        f.write(f"Party column: {party_col}\n")
        f.write(f"Documents used: {len(doc_df)}\n")
        f.write(f"Unique parties: {doc_df[party_col].nunique()}\n")
        f.write(f"Clusters: {n_clusters}\n")
        f.write(f"Method: {args.method}\n\n")

        f.write("=== Metrics ===\n")
        f.write(f"Purity: {purity:.4f}\n")
        f.write(f"ARI: {ari:.4f}\n")
        f.write(f"NMI: {nmi:.4f}\n")
        f.write(f"Silhouette: {sil:.4f}\n\n")

        f.write("=== Party Counts ===\n")
        f.write(doc_df[party_col].value_counts().to_string())
        f.write("\n")

    print("\nDone.")
    print(f"Saved clustering: {OUTPUT_CLUSTER_PATH}")
    print(f"Saved projection: {OUTPUT_PROJ_PATH}")
    print(f"Saved summary: {INFO_PATH}")


if __name__ == "__main__":
    main()
