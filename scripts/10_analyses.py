"""
Step 10 — Hypothesis tests (merges old 10 + 11 + 12)

  --hypothesis h1   : H1 — party clustering (KMeans purity / ARI / NMI / silhouette
                      in two views: raw embedding space and topic-distribution space)
  --hypothesis h3   : H3 — topic specialization by party (lift, chi-square,
                      Cramer's V, top-K specialized topics per party, log-lift heatmap)
  --hypothesis h4   : H4 — polarization by year (between-party variance,
                      mean pairwise cosine distance, topic entropy)
  --all             : run all three

Inputs:
  outputs/06_chunking_embedding/corpus_chunks.csv         (H1)
  outputs/06_chunking_embedding/chunk_embeddings.npy      (H1)
  outputs/07_bertopic/chunks_with_topics.csv              (H3)
  outputs/07_bertopic/topic_info.csv                      (H3 labels)
  outputs/09_doc_topic_vectors/doc_topic_vectors_bertopic.csv  (H1, H3, H4)
  outputs/09_doc_topic_vectors/doc_party_family.csv       (H1, H3, H4)

Outputs in outputs/10_analyses/:
  H1/  party_clustering_*.csv  + reports/party_clustering_info.txt
  H3/  topic_specialization_*.csv + reports/topic_specialization_*.txt
       + figures/topic_specialization_heatmap.png
  H4/  polarization_by_year.csv + reports/polarization_info.txt
       + figures/polarization_by_year.png
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
DOC_VEC_PATH = OUTPUTS / "09_doc_topic_vectors" / "doc_topic_vectors_bertopic.csv"

STEP_DIR = OUTPUTS / "10_analyses"
FIG_DIR = STEP_DIR / "figures"
REPORTS_DIR = STEP_DIR / "reports"
STEP_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


DOC_ID_COL = "doc_id"
EXCLUDED_FAMILIES = {"unclassified", "other"}
EPS = 1e-12


# H1 — Party clustering

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


def run_h1(args) -> None:
    out_dir = STEP_DIR / "H1"
    out_reports = REPORTS_DIR / "H1"
    out_figs = FIG_DIR / "H1"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_reports.mkdir(parents=True, exist_ok=True)
    out_figs.mkdir(parents=True, exist_ok=True)

    print("\n[H1] Loading chunks and embeddings...")
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

    print(f"[H1] View A — embedding space, {len(emb_df)} docs, k={n_clusters}")
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

    print(f"[H1] View B — topic space, {len(top_df)} docs, k={n_clusters}")
    res_top = cluster_and_score(top_vectors, top_df["party_family"].values, n_clusters, args.random_state)
    top_df["cluster"] = res_top["clusters"]
    proj_top = project_vectors(top_vectors, args.method, args.random_state)
    proj_top_df = top_df[[DOC_ID_COL, "party_family", "cluster"]].copy()
    proj_top_df["x"] = proj_top[:, 0]
    proj_top_df["y"] = proj_top[:, 1]

    top_df.to_csv(out_dir / "party_clustering_topic.csv", index=False, encoding="utf-8-sig")
    proj_top_df.to_csv(out_dir / "party_clustering_projection_topic.csv", index=False, encoding="utf-8-sig")

    # Figures: 2D projection colored by party family (View A and View B)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for fam, sub in proj_emb_df.groupby("party_family"):
        axes[0].scatter(sub["x"], sub["y"], label=fam,
                        color=PARTY_COLORS.get(fam, "gray"),
                        alpha=0.6, s=18, edgecolors="none")
    axes[0].set_title(f"View A: embedding space\n"
                      f"Purity={res_emb['purity']:.2f} ARI={res_emb['ari']:.2f} NMI={res_emb['nmi']:.2f}")
    axes[0].set_xlabel("Comp 1"); axes[0].set_ylabel("Comp 2")
    axes[0].grid(alpha=0.3)

    for fam, sub in proj_top_df.groupby("party_family"):
        axes[1].scatter(sub["x"], sub["y"], label=fam,
                        color=PARTY_COLORS.get(fam, "gray"),
                        alpha=0.6, s=18, edgecolors="none")
    axes[1].set_title(f"View B: topic distribution space\n"
                      f"Purity={res_top['purity']:.2f} ARI={res_top['ari']:.2f} NMI={res_top['nmi']:.2f}")
    axes[1].set_xlabel("Comp 1"); axes[1].set_ylabel("Comp 2")
    axes[1].grid(alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(7, len(handles)),
               fontsize=9, bbox_to_anchor=(0.5, -0.04))
    plt.suptitle(f"H1 — Party clustering ({args.method.upper()}, k={n_clusters})", y=1.02)
    plt.tight_layout()
    plt.savefig(out_figs / "h1_party_projection.png", dpi=150, bbox_inches="tight")
    plt.close()

    with open(out_reports / "party_clustering_info.txt", "w", encoding="utf-8") as f:
        f.write("=== H1 — Party Clustering ===\n\n")
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

    print(f"[H1] Done → {out_dir}")


# H3 — Topic specialization

def load_topic_labels(path: Path) -> dict[int, str]:
    if not path.exists():
        return {}
    info = pd.read_csv(path)
    if "Topic" not in info.columns or "Name" not in info.columns:
        return {}
    return dict(zip(info["Topic"].astype(int), info["Name"].astype(str)))


def run_h3(args) -> None:
    out_dir = STEP_DIR / "H3"
    out_reports = REPORTS_DIR / "H3"
    out_figs = FIG_DIR / "H3"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_reports.mkdir(parents=True, exist_ok=True)
    out_figs.mkdir(parents=True, exist_ok=True)

    print("\n[H3] Loading doc-level topic vectors...")
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

    topic_labels = load_topic_labels(TOPIC_INFO_PATH)
    if topic_labels:
        lift_long["topic_label"] = lift_long["topic_id"].map(topic_labels).fillna("")

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

    print("[H3] Computing chi2 on chunk-level (party x topic)...")
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
        f.write("=== H3 — Chi-square independence (party x topic) ===\n\n")
        f.write(f"Contingency shape: {contingency.shape}\n")
        f.write(f"Total chunks: {n}\n")
        f.write(f"Chi2 statistic: {chi2_stat:.2f}\n")
        f.write(f"Degrees of freedom: {dof}\n")
        f.write(f"p-value: {p_val:.3e}\n")
        f.write(f"Cramer's V: {cramers_v:.4f}\n")
        f.write("\n(0.1 = small, 0.3 = medium, 0.5 = large effect)\n")

    pivot = lift_long.pivot(index="party_family", columns="topic_id", values="log_lift").fillna(0.0)
    plt.figure(figsize=(max(8, 0.3 * pivot.shape[1]), max(4, 0.5 * pivot.shape[0])))
    vmax = max(abs(pivot.values.min()), abs(pivot.values.max()), 1.0)
    plt.imshow(pivot.values, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    plt.colorbar(label="log(lift) — positive = over-representation")
    plt.xticks(np.arange(pivot.shape[1]), pivot.columns, rotation=45, ha="right")
    plt.yticks(np.arange(pivot.shape[0]), pivot.index)
    plt.title("H3 — Topic specialization by party (log-lift)")
    plt.tight_layout()
    plt.savefig(out_figs / "topic_specialization_heatmap.png", dpi=150)
    plt.close()

    with open(out_reports / "topic_specialization_info.txt", "w", encoding="utf-8") as f:
        f.write("=== H3 — Topic Specialization ===\n\n")
        f.write(f"Documents used: {len(df)}\n")
        f.write(f"Topics: {len(topic_cols)}\n")
        f.write(f"Parties (families): {df['party_family'].nunique()}\n")
        f.write(f"Excluded: {sorted(EXCLUDED_FAMILIES)}\n\n")
        f.write(f"Chi2={chi2_stat:.1f}  dof={dof}  p={p_val:.3e}  Cramer's V={cramers_v:.4f}\n\n")
        f.write("=== Top-K specialized topics per party (by lift) ===\n")
        f.write(top_df.to_string(index=False))
        f.write("\n")

    print(f"[H3] Done → {out_dir}")


# H4 — Polarization by year

def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < EPS or nb < EPS:
        return float("nan")
    return 1.0 - float(np.dot(a, b) / (na * nb))


def shannon_entropy(p: np.ndarray) -> float:
    p = p[p > 0]
    return float(-np.sum(p * np.log(p))) if len(p) else 0.0


def run_h4(args) -> None:
    out_dir = STEP_DIR / "H4"
    out_reports = REPORTS_DIR / "H4"
    out_figs = FIG_DIR / "H4"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_reports.mkdir(parents=True, exist_ok=True)
    out_figs.mkdir(parents=True, exist_ok=True)

    print("\n[H4] Loading doc-level topic vectors...")
    df = pd.read_csv(DOC_VEC_PATH)
    if "year" not in df.columns or "party_family" not in df.columns:
        raise ValueError("Need 'year' and 'party_family' columns.")
    topic_cols = [c for c in df.columns if c.startswith("topic_")]
    if not topic_cols:
        raise ValueError("No topic_* columns.")
    df = df[~df["party_family"].isin(EXCLUDED_FAMILIES)].copy()
    df = df.dropna(subset=["year"])

    rows = []
    for year, sub in df.groupby("year"):
        if len(sub) < 5:
            continue
        global_mean = sub[topic_cols].values.astype(float).mean(axis=0)
        bpv_num = 0.0
        n_total = len(sub)
        party_means = {}
        for party, g in sub.groupby("party_family"):
            if len(g) < 2:
                continue
            mu_p = g[topic_cols].values.astype(float).mean(axis=0)
            party_means[party] = mu_p
            bpv_num += len(g) * np.sum((mu_p - global_mean) ** 2)
        bpv = bpv_num / n_total

        parties = list(party_means.keys())
        if len(parties) >= 2:
            dists = []
            for i in range(len(parties)):
                for j in range(i + 1, len(parties)):
                    d = cosine_distance(party_means[parties[i]], party_means[parties[j]])
                    if not np.isnan(d):
                        dists.append(d)
            mean_pairwise = float(np.mean(dists)) if dists else float("nan")
            max_pairwise = float(np.max(dists)) if dists else float("nan")
        else:
            mean_pairwise = max_pairwise = float("nan")

        corpus_dist = global_mean / (global_mean.sum() + EPS)
        entropy = shannon_entropy(corpus_dist)
        max_entropy = np.log(len(topic_cols))
        norm_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

        rows.append({
            "year": year, "n_docs": n_total, "n_parties": len(parties),
            "between_party_variance": bpv,
            "mean_pairwise_cosine_dist": mean_pairwise,
            "max_pairwise_cosine_dist": max_pairwise,
            "topic_entropy": entropy, "topic_entropy_normalized": norm_entropy,
        })

    res = pd.DataFrame(rows).sort_values("year")
    res.to_csv(out_dir / "polarization_by_year.csv", index=False, encoding="utf-8-sig")

    if not res.empty:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        years_str = res["year"].astype(str).tolist()
        axes[0].plot(years_str, res["between_party_variance"], marker="o")
        axes[0].set_title("Between-party variance\n(higher = parties differ more)")
        axes[0].set_xlabel("Year"); axes[0].set_ylabel("BPV")
        axes[1].plot(years_str, res["mean_pairwise_cosine_dist"], marker="o", label="mean")
        axes[1].plot(years_str, res["max_pairwise_cosine_dist"], marker="s", label="max", alpha=0.6)
        axes[1].set_title("Pairwise cosine distance\nbetween party centroids")
        axes[1].set_xlabel("Year"); axes[1].set_ylabel("Cosine distance"); axes[1].legend()
        axes[2].plot(years_str, res["topic_entropy_normalized"], marker="o", color="C2")
        axes[2].set_title("Topic distribution entropy\n(lower = focus on fewer topics)")
        axes[2].set_xlabel("Year"); axes[2].set_ylabel("Normalized entropy")
        for ax in axes:
            ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_figs / "polarization_by_year.png", dpi=150)
        plt.close()

    with open(out_reports / "polarization_info.txt", "w", encoding="utf-8") as f:
        f.write("=== H4 — Polarization by Year ===\n\n")
        f.write(f"Excluded families: {sorted(EXCLUDED_FAMILIES)}\n")
        f.write(f"Years analyzed: {len(res)}\n\n")
        f.write(res.to_string(index=False))
        f.write("\n\nInterpretation:\n")
        f.write("- BPV: weighted dispersion of party means around the corpus mean.\n")
        f.write("- Mean pairwise cosine: how spread apart party centroids are.\n")
        f.write("- Topic entropy: how concentrated the corpus is on few topics.\n")

    print(f"[H4] Done → {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Hypothesis tests H1, H3, H4.")
    parser.add_argument("--hypothesis", type=str, choices=["h1", "h3", "h4"], default=None)
    parser.add_argument("--all", action="store_true", help="Run all three hypotheses.")
    # H1 parameters
    parser.add_argument("--method", type=str, default="pca", choices=["pca", "tsne"])
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--min-docs", type=int, default=10)
    parser.add_argument("--top-labels", type=int, default=10)
    # H3 parameters
    parser.add_argument("--top-k", type=int, default=3, help="Top-K specialized topics per party (H3).")

    args = parser.parse_args()

    if not args.all and args.hypothesis is None:
        parser.error("Specify --hypothesis {h1,h3,h4} or --all")

    if args.all or args.hypothesis == "h1":
        run_h1(args)
    if args.all or args.hypothesis == "h3":
        run_h3(args)
    if args.all or args.hypothesis == "h4":
        run_h4(args)

    print("\nDone.")


if __name__ == "__main__":
    main()
