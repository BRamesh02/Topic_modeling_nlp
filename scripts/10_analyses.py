"""
Step 10 — Inter-family clustering and thematic specialisation.

  --analysis clustering      : KMeans (k=7) on documents in two spaces
                               (mean-embedding profile and topic-distribution
                               profile), report Purity / ARI / NMI / Silhouette.
  --analysis specialisation  : lift, chi-square, Cramer's V, top-K specialised
                               topics per family, log-lift heatmap.
  --all                      : run both.
"""

from __future__ import annotations

from pathlib import Path
from typing import List
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = PROJECT_ROOT / "outputs"

CHUNKS_PATH = OUTPUTS / "06_chunking_embedding" / "corpus_chunks.csv"
EMBEDDINGS_PATH = OUTPUTS / "06_chunking_embedding" / "chunk_embeddings.npy"
CHUNKS_TOPICS_PATH = OUTPUTS / "07_bertopic" / "chunks_with_topics.csv"
TOPIC_INFO_PATH = OUTPUTS / "07_bertopic" / "topic_info.csv"
TOPIC_LABELS_PATH = OUTPUTS / "07_bertopic" / "topic_labels.csv"
DOC_VEC_PATH = OUTPUTS / "09_doc_topic_vectors" / "doc_topic_vectors_bertopic.csv"

STEP_DIR = OUTPUTS / "10_analyses"
FIG_DIR = STEP_DIR / "figures"
REPORTS_DIR = STEP_DIR / "reports"
STEP_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


DOC_ID_COL = "doc_id"
EXCLUDED_FAMILIES = {"unclassified", "other"}


def load_topic_labels(path: Path = TOPIC_LABELS_PATH) -> dict[int, str]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if "topic_id" not in df.columns or "label" not in df.columns:
        return {}
    df = df[df["label"].notna()]
    return dict(zip(df["topic_id"].astype(int), df["label"].astype(str)))


def topic_display_name(tid: int, labels: dict[int, str] | None = None, max_len: int = 50) -> str:
    if labels and tid in labels and labels[tid]:
        label = labels[tid]
        return label if len(label) <= max_len else label[: max_len - 1] + "…"
    return f"Topic {tid}"


# Party clustering

def _safe_perplexity(n_samples: int, desired: int) -> int:
    if n_samples < 3:
        return 2
    return max(2, min(desired, (n_samples - 1) // 3))


def project_vectors(vectors: np.ndarray, method: str, random_state: int,
                    tsne_perplexity: int = 30, tsne_lr: int = 200) -> np.ndarray:
    if method == "pca":
        return PCA(n_components=2, random_state=random_state).fit_transform(vectors)
    if method == "tsne":
        return TSNE(
            n_components=2,
            perplexity=_safe_perplexity(len(vectors), tsne_perplexity),
            learning_rate=tsne_lr,
            init="pca",
            random_state=random_state,
        ).fit_transform(vectors)
    raise ValueError(f"Unknown method: {method}")


def cluster_purity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) == 0:
        return float("nan")
    table = pd.crosstab(y_pred, y_true)
    return float(table.max(axis=1).sum() / table.values.sum())


def compute_doc_embeddings(df: pd.DataFrame, embeddings: np.ndarray,
                           doc_col: str, weight_col: str | None = None
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
    return unique_docs, sums / weights_sum[:, None]


def cluster_and_score(vectors: np.ndarray, labels: np.ndarray, n_clusters: int,
                      random_state: int) -> dict:
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    pred = kmeans.fit_predict(vectors)
    purity = cluster_purity(labels, pred)
    ari = adjusted_rand_score(labels, pred) if len(set(labels)) > 1 else float("nan")
    nmi = normalized_mutual_info_score(labels, pred) if len(set(labels)) > 1 else float("nan")
    sil = float(silhouette_score(vectors, pred)) if n_clusters > 1 and len(vectors) > n_clusters else float("nan")
    return {"clusters": pred, "purity": purity, "ari": ari, "nmi": nmi, "silhouette": sil}


PARTY_COLORS = {
    "radical_left":         "#7a0000",
    "communist_left":       "#c40000",
    "socialist_left":       "#ff6961",
    "ecologist":            "#1ea672",
    "liberal_center_right": "#fbbf24",
    "gaullist_right":       "#1e40af",
    "national_right":       "#000000",
}


def run_clustering(args) -> None:
    out_dir = STEP_DIR / "clustering"
    out_reports = REPORTS_DIR / "clustering"
    out_figs = FIG_DIR / "clustering"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_reports.mkdir(parents=True, exist_ok=True)
    out_figs.mkdir(parents=True, exist_ok=True)

    print("\n[clustering] Loading chunks and embeddings...")
    df_chunks = pd.read_csv(CHUNKS_PATH)
    embeddings = np.load(EMBEDDINGS_PATH)
    if len(df_chunks) != embeddings.shape[0]:
        raise ValueError("Mismatch between chunks and embeddings.")

    doc_topic = pd.read_csv(DOC_VEC_PATH)
    doc_topic[DOC_ID_COL] = doc_topic[DOC_ID_COL].astype(str)

    if "party_family" not in doc_topic.columns:
        raise ValueError("party_family missing — run 09_doc_topic_vectors.py first.")

    df_chunks[DOC_ID_COL] = df_chunks[DOC_ID_COL].astype(str)
    doc_ids_emb, doc_embeddings = compute_doc_embeddings(df_chunks, embeddings, DOC_ID_COL)

    emb_df = pd.DataFrame({DOC_ID_COL: doc_ids_emb})
    emb_df = emb_df.merge(doc_topic[[DOC_ID_COL, "party_family"]], on=DOC_ID_COL, how="left")
    emb_df["party_family"] = emb_df["party_family"].fillna("Unknown")

    counts = emb_df["party_family"].value_counts()
    allowed = counts[counts >= args.min_docs]
    allowed = allowed[~allowed.index.isin(EXCLUDED_FAMILIES)]
    if args.top_labels and args.top_labels > 0:
        allowed = allowed.head(args.top_labels)

    mask = emb_df["party_family"].isin(allowed.index).values
    emb_df = emb_df.loc[mask].reset_index(drop=True)
    doc_embeddings = doc_embeddings[mask]
    keep_doc_ids = set(emb_df[DOC_ID_COL].tolist())

    n_clusters = emb_df["party_family"].nunique()
    labels_emb = emb_df["party_family"].values

    print(f"[clustering] View A — embedding space, {len(emb_df)} docs, k={n_clusters}")
    res_emb = cluster_and_score(doc_embeddings, labels_emb, n_clusters, args.random_state)
    emb_df["cluster"] = res_emb["clusters"]
    proj_emb = project_vectors(doc_embeddings, args.method, args.random_state)
    proj_emb_df = emb_df[[DOC_ID_COL, "party_family", "cluster"]].copy()
    proj_emb_df["x"] = proj_emb[:, 0]
    proj_emb_df["y"] = proj_emb[:, 1]

    emb_df.to_csv(out_dir / "party_clustering_embedding.csv", index=False, encoding="utf-8-sig")
    proj_emb_df.to_csv(out_dir / "party_clustering_projection_embedding.csv", index=False, encoding="utf-8-sig")

    topic_cols = [c for c in doc_topic.columns if c.startswith("topic_")]
    if not topic_cols:
        raise ValueError("No topic_* columns in doc_topic_vectors_bertopic.csv")
    top_df = doc_topic[doc_topic[DOC_ID_COL].isin(keep_doc_ids)].copy().reset_index(drop=True)
    top_vectors = top_df[topic_cols].values.astype(float)

    print(f"[clustering] View B — topic space, {len(top_df)} docs, k={n_clusters}")
    res_top = cluster_and_score(top_vectors, top_df["party_family"].values, n_clusters, args.random_state)
    top_df["cluster"] = res_top["clusters"]
    proj_top = project_vectors(top_vectors, args.method, args.random_state)
    proj_top_df = top_df[[DOC_ID_COL, "party_family", "cluster"]].copy()
    proj_top_df["x"] = proj_top[:, 0]
    proj_top_df["y"] = proj_top[:, 1]

    top_df.to_csv(out_dir / "party_clustering_topic.csv", index=False, encoding="utf-8-sig")
    proj_top_df.to_csv(out_dir / "party_clustering_projection_topic.csv", index=False, encoding="utf-8-sig")

    from umap import UMAP as _UMAP_VIZ

    chunks_with_topic_path = OUTPUTS / "07_bertopic" / "chunks_with_topics.csv"
    chunks_topic_df = pd.read_csv(chunks_with_topic_path, usecols=["doc_id", "topic"])
    chunks_topic_df["doc_id"] = chunks_topic_df["doc_id"].astype(str)
    chunk_in_seven = (
        chunks_topic_df["doc_id"].isin(set(emb_df[DOC_ID_COL]))
        & (chunks_topic_df["topic"] != -1)
    )
    chunk_idx = np.where(chunk_in_seven.values)[0]
    chunk_emb = embeddings[chunk_idx]
    chunk_meta = chunks_topic_df.iloc[chunk_idx].reset_index(drop=True)
    chunk_meta = chunk_meta.merge(
        doc_topic[[DOC_ID_COL, "party_family"]], on=DOC_ID_COL, how="left"
    )

    print(f"[clustering] Running UMAP on {len(chunk_emb)} chunks for visualisation...")
    chunk_xy = _UMAP_VIZ(
        n_components=2, n_neighbors=30, min_dist=0.3,
        metric="cosine", random_state=args.random_state,
    ).fit_transform(chunk_emb)
    chunk_meta["x"] = chunk_xy[:, 0]
    chunk_meta["y"] = chunk_xy[:, 1]

    panels = [
        ("Peripheries", ["national_right", "ecologist", "radical_left"]),
        ("Mainstream",  ["socialist_left", "gaullist_right", "liberal_center_right"]),
        ("FN vs écologistes", ["national_right", "ecologist"]),
    ]

    xlim = (np.percentile(chunk_meta["x"], 2),  np.percentile(chunk_meta["x"], 98))
    ylim = (np.percentile(chunk_meta["y"], 2),  np.percentile(chunk_meta["y"], 98))

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5), sharex=True, sharey=True)

    panel_cx = (xlim[0] + xlim[1]) / 2
    panel_cy = (ylim[0] + ylim[1]) / 2
    panel_radius = 0.5 * max(xlim[1] - xlim[0], ylim[1] - ylim[0])

    for ax, (title, focus_fams) in zip(axes, panels):
        bg = chunk_meta[~chunk_meta["party_family"].isin(focus_fams)]
        ax.scatter(bg["x"], bg["y"], s=3, alpha=0.15,
                   color="#cccccc", edgecolors="none", rasterized=True)

        annotations = []
        for k, fam in enumerate(focus_fams):
            sub = chunk_meta[chunk_meta["party_family"] == fam]
            ax.scatter(sub["x"], sub["y"], s=6, alpha=0.6,
                       color=PARTY_COLORS.get(fam, "tab:blue"),
                       label=f"{fam} (n={len(sub)})",
                       edgecolors="none", rasterized=True)
            cx, cy = sub["x"].median(), sub["y"].median()
            dx, dy = cx - panel_cx, cy - panel_cy
            norm = (dx ** 2 + dy ** 2) ** 0.5
            if norm < 0.1:
                angle = (2 * np.pi * k) / max(1, len(focus_fams)) + np.pi / 4
                dx, dy = np.cos(angle), np.sin(angle)
            else:
                dx, dy = dx / norm, dy / norm
            label_x = cx + dx * panel_radius * 0.55
            label_y = cy + dy * panel_radius * 0.55
            annotations.append((fam, cx, cy, label_x, label_y))

        for fam, cx, cy, lx, ly in annotations:
            color = PARTY_COLORS.get(fam, "gray")
            ax.annotate(
                fam,
                xy=(cx, cy), xytext=(lx, ly),
                ha="center", va="center",
                fontsize=11, fontweight="bold", color="white",
                bbox=dict(boxstyle="round,pad=0.3", facecolor=color,
                          edgecolor="black", linewidth=0.6, alpha=0.95),
                arrowprops=dict(arrowstyle="-", color=color, linewidth=1.2,
                                shrinkA=4, shrinkB=4, alpha=0.85),
            )
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.set_xticks([]); ax.set_yticks([])
        ax.legend(loc="lower left", fontsize=10, frameon=True, markerscale=2.5)

    plt.tight_layout()
    plt.savefig(out_figs / "party_projection.png", dpi=150, bbox_inches="tight")
    plt.close()

    with open(out_reports / "party_clustering_info.txt", "w", encoding="utf-8") as f:
        f.write("=== Party clustering ===\n\n")
        f.write(f"Excluded families: {sorted(EXCLUDED_FAMILIES)}\n")
        f.write(f"Documents used: {len(emb_df)}\n")
        f.write(f"Number of unique labels: {emb_df['party_family'].nunique()}\n")
        f.write(f"K (clusters): {n_clusters}\n")
        f.write(f"Projection: {args.method}\n\n")

        f.write("=== View A: Raw embedding space ===\n")
        f.write(f"Purity:     {res_emb['purity']:.4f}\n")
        f.write(f"ARI:        {res_emb['ari']:.4f}\n")
        f.write(f"NMI:        {res_emb['nmi']:.4f}\n")
        f.write(f"Silhouette: {res_emb['silhouette']:.4f}\n\n")

        f.write("=== View B: Topic distribution space ===\n")
        f.write(f"Purity:     {res_top['purity']:.4f}\n")
        f.write(f"ARI:        {res_top['ari']:.4f}\n")
        f.write(f"NMI:        {res_top['nmi']:.4f}\n")
        f.write(f"Silhouette: {res_top['silhouette']:.4f}\n\n")

        f.write("=== Label counts ===\n")
        f.write(emb_df["party_family"].value_counts().to_string())
        f.write("\n")

    print(f"[clustering] Done → {out_dir}")


# Topic specialization

def load_topic_auto_names(path: Path) -> dict[int, str]:
    if not path.exists():
        return {}
    info = pd.read_csv(path)
    if "Topic" not in info.columns or "Name" not in info.columns:
        return {}
    return dict(zip(info["Topic"].astype(int), info["Name"].astype(str)))


def run_specialisation(args) -> None:
    out_dir = STEP_DIR / "specialisation"
    out_reports = REPORTS_DIR / "specialisation"
    out_figs = FIG_DIR / "specialisation"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_reports.mkdir(parents=True, exist_ok=True)
    out_figs.mkdir(parents=True, exist_ok=True)

    print("\n[specialisation] Loading doc-level topic vectors...")
    doc_vec = pd.read_csv(DOC_VEC_PATH)
    if "party_family" not in doc_vec.columns:
        raise ValueError("party_family missing — run 09_doc_topic_vectors.py first.")
    topic_cols = [c for c in doc_vec.columns if c.startswith("topic_")]
    if not topic_cols:
        raise ValueError("No topic_* columns in doc-level vectors.")

    df = doc_vec[~doc_vec["party_family"].isin(EXCLUDED_FAMILIES)].copy()
    if df.empty:
        raise ValueError("No documents left after excluding noise families.")

    party_share = df.groupby("party_family")[topic_cols].mean()
    global_share = df[topic_cols].mean()
    lift = party_share.div(global_share.replace(0, np.nan), axis=1).fillna(0.0)

    lift_long = lift.reset_index().melt(id_vars="party_family", var_name="topic", value_name="lift")
    lift_long["topic_id"] = lift_long["topic"].str.replace("topic_", "").astype(int)
    lift_long["log_lift"] = np.log(np.maximum(lift_long["lift"].values, 1e-9))

    auto_names = load_topic_auto_names(TOPIC_INFO_PATH)
    human_labels = load_topic_labels(TOPIC_LABELS_PATH)
    if auto_names:
        lift_long["topic_auto_name"] = lift_long["topic_id"].map(auto_names).fillna("")
    if human_labels:
        lift_long["topic_label"] = lift_long["topic_id"].map(human_labels).fillna("")

    lift_long.to_csv(out_dir / "topic_specialization_lift.csv", index=False, encoding="utf-8-sig")

    rows = []
    for party, sub in lift_long.groupby("party_family"):
        top_k = sub.sort_values("lift", ascending=False).head(args.top_k)
        for rank, (_, r) in enumerate(top_k.iterrows(), start=1):
            rows.append({
                "party_family": party,
                "rank": rank,
                "topic_id": r["topic_id"],
                "topic_label": r.get("topic_label", ""),
                "lift": r["lift"],
                "log_lift": r["log_lift"],
            })
    top_df = pd.DataFrame(rows)
    top_df.to_csv(out_dir / "topic_specialization_top.csv", index=False, encoding="utf-8-sig")

    print("[specialisation] Computing chi2 on chunk-level (party x topic)...")
    chunks = pd.read_csv(CHUNKS_TOPICS_PATH)
    chunks = chunks[chunks["topic"] != -1].copy()
    party_map = dict(zip(doc_vec[DOC_ID_COL].astype(str), doc_vec["party_family"]))
    chunks[DOC_ID_COL] = chunks[DOC_ID_COL].astype(str)
    chunks["party_family"] = chunks[DOC_ID_COL].map(party_map).fillna("unclassified")
    chunks = chunks[~chunks["party_family"].isin(EXCLUDED_FAMILIES)]

    contingency = pd.crosstab(chunks["party_family"], chunks["topic"])
    chi2_stat, p_val, dof, _ = chi2_contingency(contingency)
    n = contingency.values.sum()
    cramers_v = float(np.sqrt(chi2_stat / (n * (min(contingency.shape) - 1))))

    with open(out_reports / "topic_specialization_chi2.txt", "w", encoding="utf-8") as f:
        f.write("=== Chi-square independence (party x topic) ===\n\n")
        f.write(f"Contingency shape: {contingency.shape}\n")
        f.write(f"Total chunks: {n}\n")
        f.write(f"Chi2 statistic: {chi2_stat:.2f}\n")
        f.write(f"Degrees of freedom: {dof}\n")
        f.write(f"p-value: {p_val:.3e}\n")
        f.write(f"Cramer's V: {cramers_v:.4f}\n")
        f.write("\n(0.1 = small, 0.3 = medium, 0.5 = large effect)\n")

    pivot = lift_long.pivot(index="party_family", columns="topic_id", values="log_lift").fillna(0.0)
    pivot_display = pivot.clip(lower=-3.0, upper=3.0)
    xtick_labels = [topic_display_name(int(t), human_labels, max_len=35) for t in pivot.columns]
    plt.figure(figsize=(max(16, 0.7 * pivot.shape[1]), max(8, 1.0 * pivot.shape[0])))
    plt.imshow(pivot_display.values, aspect="auto", cmap="RdBu_r", vmin=-3, vmax=3)
    cbar = plt.colorbar(label="log(lift) — positive = over-representation (clipped at ±3)")
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label("log(lift) — positive = over-representation (clipped at ±3)", fontsize=13)
    plt.xticks(np.arange(pivot.shape[1]), xtick_labels, rotation=60, ha="right", fontsize=13)
    plt.yticks(np.arange(pivot.shape[0]), pivot.index, fontsize=14)
    plt.title("Topic specialisation by party (log-lift)", fontsize=16)
    plt.tight_layout()
    plt.savefig(out_figs / "topic_specialization_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()

    with open(out_reports / "topic_specialization_info.txt", "w", encoding="utf-8") as f:
        f.write("=== Topic specialisation ===\n\n")
        f.write(f"Documents used: {len(df)}\n")
        f.write(f"Topics: {len(topic_cols)}\n")
        f.write(f"Parties (families): {df['party_family'].nunique()}\n")
        f.write(f"Excluded: {sorted(EXCLUDED_FAMILIES)}\n\n")
        f.write(f"Chi2={chi2_stat:.1f}  dof={dof}  p={p_val:.3e}  Cramer's V={cramers_v:.4f}\n\n")
        f.write("=== Top-K specialized topics per party (by lift) ===\n")
        f.write(top_df.to_string(index=False))
        f.write("\n")

    print(f"[specialisation] Done → {out_dir}")




def main():
    parser = argparse.ArgumentParser(description="Inter-family clustering and thematic specialisation.")
    parser.add_argument("--analysis", type=str, choices=["clustering", "specialisation"], default=None)
    parser.add_argument("--all", action="store_true", help="Run both analyses.")
    # Clustering parameters
    parser.add_argument("--method", type=str, default="pca", choices=["pca", "tsne"])
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--min-docs", type=int, default=10)
    parser.add_argument("--top-labels", type=int, default=10)
    # Specialisation parameters
    parser.add_argument("--top-k", type=int, default=3, help="Top-K specialised topics per party.")

    args = parser.parse_args()

    if not args.all and args.analysis is None:
        parser.error("Specify --analysis {clustering,specialisation} or --all")

    if args.all or args.analysis == "clustering":
        run_clustering(args)
    if args.all or args.analysis == "specialisation":
        run_specialisation(args)

    print("\nDone.")


if __name__ == "__main__":
    main()
