"""
Step 10 — Inter-family clustering and thematic specialisation.

  Clustering:    KMeans (k=7) on documents in two spaces (mean embedding +
                 topic distribution). Reports Purity / ARI / NMI / Silhouette
                 and a UMAP projection of the chunks.
  Specialisation: Lift, chi2, Cramer's V on the family x topic contingency,
                  plus top-3 specialised topics per family and a log-lift
                  heatmap.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)


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
RANDOM_STATE = 42
TOP_K = 3

PARTY_COLORS = {
    "radical_left":         "#7a0000",
    "communist_left":       "#c40000",
    "socialist_left":       "#ff6961",
    "ecologist":            "#1ea672",
    "liberal_center_right": "#fbbf24",
    "gaullist_right":       "#1e40af",
    "national_right":       "#000000",
}


def load_topic_labels():
    if not TOPIC_LABELS_PATH.exists():
        return {}
    df = pd.read_csv(TOPIC_LABELS_PATH)
    df = df[df["label"].notna()]
    return dict(zip(df["topic_id"].astype(int), df["label"].astype(str)))


def topic_display_name(tid, labels, max_len=35):
    label = labels.get(int(tid), "")
    if not label:
        return f"Topic {tid}"
    return label if len(label) <= max_len else label[: max_len - 1] + "…"


def cluster_purity(y_true, y_pred):
    table = pd.crosstab(y_pred, y_true)
    return float(table.max(axis=1).sum() / table.values.sum())


def doc_mean_embeddings(df, embeddings):
    doc_ids = df[DOC_ID_COL].astype(str).values
    unique_docs, inverse = np.unique(doc_ids, return_inverse=True)
    sums = np.zeros((len(unique_docs), embeddings.shape[1]), dtype=float)
    counts = np.zeros(len(unique_docs), dtype=float)
    np.add.at(sums, inverse, embeddings)
    np.add.at(counts, inverse, 1.0)
    return unique_docs, sums / counts[:, None]


def kmeans_score(vectors, labels, n_clusters):
    pred = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init="auto").fit_predict(vectors)
    return {
        "clusters": pred,
        "purity": cluster_purity(labels, pred),
        "ari": adjusted_rand_score(labels, pred),
        "nmi": normalized_mutual_info_score(labels, pred),
        "silhouette": float(silhouette_score(vectors, pred)),
    }


def run_clustering():
    out_dir = STEP_DIR / "clustering"
    out_reports = REPORTS_DIR / "clustering"
    out_figs = FIG_DIR / "clustering"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_reports.mkdir(parents=True, exist_ok=True)
    out_figs.mkdir(parents=True, exist_ok=True)

    print("\n[clustering] Loading chunks and embeddings...")
    df_chunks = pd.read_csv(CHUNKS_PATH)
    embeddings = np.load(EMBEDDINGS_PATH)
    df_chunks[DOC_ID_COL] = df_chunks[DOC_ID_COL].astype(str)

    doc_topic = pd.read_csv(DOC_VEC_PATH)
    doc_topic[DOC_ID_COL] = doc_topic[DOC_ID_COL].astype(str)

    doc_ids_emb, doc_embeddings = doc_mean_embeddings(df_chunks, embeddings)
    emb_df = pd.DataFrame({DOC_ID_COL: doc_ids_emb})
    emb_df = emb_df.merge(doc_topic[[DOC_ID_COL, "party_family"]], on=DOC_ID_COL, how="left")
    emb_df["party_family"] = emb_df["party_family"].fillna("Unknown")

    mask = ~emb_df["party_family"].isin(EXCLUDED_FAMILIES | {"Unknown"})
    emb_df = emb_df.loc[mask].reset_index(drop=True)
    doc_embeddings = doc_embeddings[mask.values]
    keep_doc_ids = set(emb_df[DOC_ID_COL])

    n_clusters = emb_df["party_family"].nunique()
    labels_emb = emb_df["party_family"].values

    print(f"[clustering] View A — embedding space, {len(emb_df)} docs, k={n_clusters}")
    res_emb = kmeans_score(doc_embeddings, labels_emb, n_clusters)
    emb_df["cluster"] = res_emb["clusters"]
    emb_df.to_csv(out_dir / "party_clustering_embedding.csv", index=False, encoding="utf-8-sig")

    topic_cols = [c for c in doc_topic.columns if c.startswith("topic_")]
    top_df = doc_topic[doc_topic[DOC_ID_COL].isin(keep_doc_ids)].copy().reset_index(drop=True)
    top_vectors = top_df[topic_cols].values.astype(float)

    print(f"[clustering] View B — topic space, {len(top_df)} docs, k={n_clusters}")
    res_top = kmeans_score(top_vectors, top_df["party_family"].values, n_clusters)
    top_df["cluster"] = res_top["clusters"]
    top_df.to_csv(out_dir / "party_clustering_topic.csv", index=False, encoding="utf-8-sig")

    # UMAP figure (3 panels)
    from umap import UMAP

    chunks_topic_df = pd.read_csv(CHUNKS_TOPICS_PATH, usecols=["doc_id", "topic"])
    chunks_topic_df["doc_id"] = chunks_topic_df["doc_id"].astype(str)
    keep = chunks_topic_df["doc_id"].isin(keep_doc_ids) & (chunks_topic_df["topic"] != -1)
    chunk_idx = np.where(keep.values)[0]
    chunk_emb = embeddings[chunk_idx]
    chunk_meta = chunks_topic_df.iloc[chunk_idx].reset_index(drop=True)
    chunk_meta = chunk_meta.merge(doc_topic[[DOC_ID_COL, "party_family"]], on=DOC_ID_COL, how="left")

    print(f"[clustering] Running UMAP on {len(chunk_emb)} chunks for visualisation...")
    chunk_xy = UMAP(n_components=2, n_neighbors=30, min_dist=0.3,
                    metric="cosine", random_state=RANDOM_STATE).fit_transform(chunk_emb)
    chunk_meta["x"] = chunk_xy[:, 0]
    chunk_meta["y"] = chunk_xy[:, 1]

    panels = [
        ("Peripheries", ["national_right", "ecologist", "radical_left"]),
        ("Mainstream",  ["socialist_left", "gaullist_right", "liberal_center_right"]),
        ("FN vs écologistes", ["national_right", "ecologist"]),
    ]

    xlim = (np.percentile(chunk_meta["x"], 2), np.percentile(chunk_meta["x"], 98))
    ylim = (np.percentile(chunk_meta["y"], 2), np.percentile(chunk_meta["y"], 98))
    panel_cx = (xlim[0] + xlim[1]) / 2
    panel_cy = (ylim[0] + ylim[1]) / 2
    panel_radius = 0.5 * max(xlim[1] - xlim[0], ylim[1] - ylim[0])

    _, axes = plt.subplots(1, 3, figsize=(18, 6.5), sharex=True, sharey=True)

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
            annotations.append((fam, cx, cy,
                                cx + dx * panel_radius * 0.55,
                                cy + dy * panel_radius * 0.55))

        for fam, cx, cy, lx, ly in annotations:
            color = PARTY_COLORS.get(fam, "gray")
            ax.annotate(
                fam, xy=(cx, cy), xytext=(lx, ly),
                ha="center", va="center",
                fontsize=11, fontweight="bold", color="white",
                bbox=dict(boxstyle="round,pad=0.3", facecolor=color,
                          edgecolor="black", linewidth=0.6, alpha=0.95),
                arrowprops=dict(arrowstyle="-", color=color, linewidth=1.2,
                                shrinkA=4, shrinkB=4, alpha=0.85),
            )

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(loc="lower left", fontsize=10, frameon=True, markerscale=2.5)

    plt.tight_layout()
    plt.savefig(out_figs / "party_projection.png", dpi=150, bbox_inches="tight")
    plt.close()

    with open(out_reports / "party_clustering_info.txt", "w", encoding="utf-8") as f:
        f.write("=== Party clustering ===\n\n")
        f.write(f"Documents used: {len(emb_df)}\n")
        f.write(f"K (clusters): {n_clusters}\n\n")
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


def run_specialisation():
    out_dir = STEP_DIR / "specialisation"
    out_reports = REPORTS_DIR / "specialisation"
    out_figs = FIG_DIR / "specialisation"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_reports.mkdir(parents=True, exist_ok=True)
    out_figs.mkdir(parents=True, exist_ok=True)

    print("\n[specialisation] Loading doc-level topic vectors...")
    doc_vec = pd.read_csv(DOC_VEC_PATH)
    topic_cols = [c for c in doc_vec.columns if c.startswith("topic_")]

    df = doc_vec[~doc_vec["party_family"].isin(EXCLUDED_FAMILIES)].copy()

    party_share = df.groupby("party_family")[topic_cols].mean()
    global_share = df[topic_cols].mean()
    lift = party_share.div(global_share.replace(0, np.nan), axis=1).fillna(0.0)

    lift_long = lift.reset_index().melt(id_vars="party_family", var_name="topic", value_name="lift")
    lift_long["topic_id"] = lift_long["topic"].str.replace("topic_", "").astype(int)
    lift_long["log_lift"] = np.log(np.maximum(lift_long["lift"].values, 1e-9))

    human_labels = load_topic_labels()
    if human_labels:
        lift_long["topic_label"] = lift_long["topic_id"].map(human_labels).fillna("")

    lift_long.to_csv(out_dir / "topic_specialization_lift.csv", index=False, encoding="utf-8-sig")

    rows = []
    for party, sub in lift_long.groupby("party_family"):
        for rank, (_, r) in enumerate(sub.sort_values("lift", ascending=False).head(TOP_K).iterrows(), start=1):
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
    xtick_labels = [topic_display_name(t, human_labels) for t in pivot.columns]

    plt.figure(figsize=(max(16, 0.7 * pivot.shape[1]), max(8, 1.0 * pivot.shape[0])))
    plt.imshow(pivot_display.values, aspect="auto", cmap="RdBu_r", vmin=-3, vmax=3)
    cbar = plt.colorbar()
    cbar.set_label("log(lift) — positive = over-representation (clipped at ±3)", fontsize=13)
    cbar.ax.tick_params(labelsize=12)
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
        f.write(f"Parties (families): {df['party_family'].nunique()}\n\n")
        f.write(f"Chi2={chi2_stat:.1f}  dof={dof}  p={p_val:.3e}  Cramer's V={cramers_v:.4f}\n\n")
        f.write("=== Top-K specialized topics per party (by lift) ===\n")
        f.write(top_df.to_string(index=False))
        f.write("\n")

    print(f"[specialisation] Done → {out_dir}")


def main():
    run_clustering()
    run_specialisation()
    print("\nDone.")


if __name__ == "__main__":
    main()
