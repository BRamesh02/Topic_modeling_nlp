"""
Step 13 — Topic positioning over time for parties.

Compute per-party, per-year centroid embeddings restricted to a topic,
project to 2D, and plot trajectories to inspect shifts over election years.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from _family_mapping import assign_party_family


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = PROJECT_ROOT / "outputs"

CHUNKS_TOPICS_PATH = OUTPUTS / "07_bertopic" / "chunks_with_topics.csv"
EMBEDDINGS_PATH = OUTPUTS / "06_chunking_embedding" / "chunk_embeddings.npy"
TOPIC_INFO_PATH = OUTPUTS / "07_bertopic" / "topic_info.csv"
TOPIC_LABELS_PATH = OUTPUTS / "07_bertopic" / "topic_labels.csv"

STEP_DIR = OUTPUTS / "13_topic_positioning"
DATA_DIR = STEP_DIR / "data"
FIG_DIR = STEP_DIR / "figures"
REPORTS_DIR = STEP_DIR / "reports"

STEP_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

DOC_ID_COL = "doc_id"
TOPIC_COL = "topic"
YEAR_COL = "year"

EXCLUDED_PARTY_LABELS = {"", "unclassified", "other"}

DEFAULT_META_CANDIDATES = [
    PROJECT_ROOT / "data" / "archelect_search.csv",
    PROJECT_ROOT.parent / "archelect_search.csv",
]


def find_metadata_path() -> Path | None:
    for path in DEFAULT_META_CANDIDATES:
        if path.exists():
            return path
    return None


def load_metadata(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    df.columns = [c.strip().lower() for c in df.columns]
    if "id" not in df.columns:
        raise ValueError("Metadata CSV missing 'id' column.")
    if "date" in df.columns and "year" not in df.columns:
        df["year"] = df["date"].astype(str).str[:4]
    cols = [c for c in [
        "id",
        "date",
        "year",
        "titulaire-soutien",
        "titulaire-liste",
    ] if c in df.columns]
    return df[cols].drop_duplicates("id")


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value)).strip().lower()


def load_topic_labels(path: Path = TOPIC_LABELS_PATH) -> dict[int, str]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if "topic_id" not in df.columns or "label" not in df.columns:
        return {}
    df = df[df["label"].notna()]
    return dict(zip(df["topic_id"].astype(int), df["label"].astype(str)))


def load_topic_auto_names(path: Path = TOPIC_INFO_PATH) -> dict[int, str]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if "Topic" not in df.columns or "Name" not in df.columns:
        return {}
    return dict(zip(df["Topic"].astype(int), df["Name"].astype(str)))


def topic_display_name(tid: int, labels: dict[int, str], auto: dict[int, str]) -> str:
    if tid in labels and labels[tid]:
        return labels[tid]
    if tid in auto and auto[tid]:
        return auto[tid]
    return f"Topic {tid}"


def resolve_topic_ids(
    queries: list[str],
    labels: dict[int, str],
    auto: dict[int, str],
    available: set[int],
) -> tuple[list[int], list[str]]:
    resolved = []
    notes = []
    for q in queries:
        q = str(q).strip()
        if not q:
            continue
        if q.isdigit() or (q.startswith("-") and q[1:].isdigit()):
            tid = int(q)
            if tid in available:
                resolved.append(tid)
            else:
                notes.append(f"Topic id {tid} not found in data.")
            continue

        q_norm = normalize_text(q)
        matches = []
        for tid in available:
            label = normalize_text(labels.get(tid, ""))
            name = normalize_text(auto.get(tid, ""))
            if q_norm in label or q_norm in name:
                matches.append(tid)
        if matches:
            resolved.extend(matches)
        else:
            notes.append(f"No topic match for '{q}'.")

    resolved = sorted(set(resolved))
    return resolved, notes


def safe_perplexity(n_samples: int, desired: int) -> int:
    if n_samples < 3:
        return 2
    return max(2, min(desired, (n_samples - 1) // 3))


def project_2d(vectors: np.ndarray, method: str, random_state: int) -> np.ndarray:
    method = method.lower()
    if method == "pca":
        from sklearn.decomposition import PCA

        return PCA(n_components=2, random_state=random_state).fit_transform(vectors)
    if method == "tsne":
        from sklearn.manifold import TSNE

        return TSNE(
            n_components=2,
            perplexity=safe_perplexity(len(vectors), 30),
            learning_rate=200,
            init="pca",
            random_state=random_state,
        ).fit_transform(vectors)
    if method == "umap":
        try:
            from umap import UMAP
        except Exception as exc:
            raise RuntimeError("UMAP is not available; install umap-learn or use PCA/TSNE.") from exc
        n_neighbors = min(15, max(2, len(vectors) - 1))
        return UMAP(
            n_neighbors=n_neighbors,
            n_components=2,
            min_dist=0.1,
            metric="cosine",
            random_state=random_state,
        ).fit_transform(vectors)
    raise ValueError(f"Unknown method: {method}")


def build_party_label(df: pd.DataFrame, mode: str) -> pd.Series:
    mode = mode.lower()
    if mode == "family":
        return df.apply(assign_party_family, axis=1)
    if mode == "soutien":
        return df.get("titulaire-soutien", "").fillna("").astype(str)
    if mode == "liste":
        return df.get("titulaire-liste", "").fillna("").astype(str)
    raise ValueError("party-mode must be one of: family, soutien, liste")


def main() -> None:
    parser = argparse.ArgumentParser(description="Topic positioning over time for parties.")
    parser.add_argument("--topics", nargs="*", default=[],
                        help="Topic ids or name fragments (label or auto-name).")
    parser.add_argument("--top-topics", type=int, default=5,
                        help="If --topics is empty, use the top-N topics by size.")
    parser.add_argument("--party-mode", type=str, default="family",
                        help="Party label source: family | soutien | liste.")
    parser.add_argument("--parties", nargs="*", default=[],
                        help="List of party labels to keep (case-insensitive).")
    parser.add_argument("--top-parties", type=int, default=6,
                        help="If --parties empty, keep top-N parties by chunk count.")
    parser.add_argument("--min-chunks", type=int, default=12,
                        help="Minimum chunks per (party, year, topic) cell.")
    parser.add_argument("--method", type=str, default="pca",
                        help="2D projection: pca | tsne | umap.")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--meta-path", type=str, default=None,
                        help="Optional path to archelect_search.csv.")
    args = parser.parse_args()

    if not CHUNKS_TOPICS_PATH.exists():
        raise FileNotFoundError(f"Missing {CHUNKS_TOPICS_PATH}. Run 07_bertopic.py first.")
    if not EMBEDDINGS_PATH.exists():
        raise FileNotFoundError(f"Missing {EMBEDDINGS_PATH}. Run 06_chunking_embedding.py first.")

    print("Loading chunks with topics...")
    chunks = pd.read_csv(CHUNKS_TOPICS_PATH)
    if DOC_ID_COL not in chunks.columns or TOPIC_COL not in chunks.columns:
        raise ValueError("Missing required columns in chunks file.")

    print("Loading embeddings...")
    embeddings = np.load(EMBEDDINGS_PATH)
    if len(chunks) != embeddings.shape[0]:
        raise ValueError("Mismatch between chunks and embeddings.")

    chunks = chunks.copy()
    chunks["chunk_index"] = np.arange(len(chunks))

    meta_path = Path(args.meta_path) if args.meta_path else find_metadata_path()
    if meta_path and meta_path.exists():
        print(f"Loading metadata from {meta_path}...")
        meta = load_metadata(meta_path)
        meta = meta.rename(columns={"id": DOC_ID_COL})
        chunks = chunks.merge(meta, on=DOC_ID_COL, how="left", suffixes=("", "_meta"))

        if "year_meta" in chunks.columns:
            if YEAR_COL in chunks.columns:
                chunks[YEAR_COL] = chunks[YEAR_COL].fillna(chunks["year_meta"])
            else:
                chunks[YEAR_COL] = chunks["year_meta"]
        if "titulaire-soutien_meta" in chunks.columns:
            if "titulaire-soutien" in chunks.columns:
                chunks["titulaire-soutien"] = chunks["titulaire-soutien"].fillna(
                    chunks["titulaire-soutien_meta"]
                )
            else:
                chunks["titulaire-soutien"] = chunks["titulaire-soutien_meta"]
        if "titulaire-liste_meta" in chunks.columns:
            if "titulaire-liste" in chunks.columns:
                chunks["titulaire-liste"] = chunks["titulaire-liste"].fillna(
                    chunks["titulaire-liste_meta"]
                )
            else:
                chunks["titulaire-liste"] = chunks["titulaire-liste_meta"]

    if YEAR_COL not in chunks.columns:
        raise ValueError("Year column missing. Ensure metadata is available.")

    chunks[TOPIC_COL] = pd.to_numeric(chunks[TOPIC_COL], errors="coerce")
    chunks[YEAR_COL] = chunks[YEAR_COL].astype(str).str[:4]
    chunks["party_label"] = build_party_label(chunks, args.party_mode)
    chunks["party_label"] = chunks["party_label"].fillna("").astype(str)

    labels = load_topic_labels()
    auto_names = load_topic_auto_names()
    available_topics = set(chunks[TOPIC_COL].dropna().astype(int).unique())
    available_topics.discard(-1)

    if args.topics:
        topic_ids, notes = resolve_topic_ids(args.topics, labels, auto_names, available_topics)
    else:
        topic_ids = []
        notes = []

    if not topic_ids:
        info = pd.read_csv(TOPIC_INFO_PATH) if TOPIC_INFO_PATH.exists() else None
        if info is None or "Topic" not in info.columns:
            topic_ids = sorted(list(available_topics))
        else:
            info = info[info["Topic"] != -1].sort_values("Count", ascending=False)
            topic_ids = info["Topic"].astype(int).head(args.top_topics).tolist()
        notes.append(f"Using top-{len(topic_ids)} topics by size: {topic_ids}.")

    report_lines = []
    report_lines.append("=== Topic Positioning ===")
    for note in notes:
        report_lines.append(note)

    all_rows = []

    for tid in topic_ids:
        sub = chunks[chunks[TOPIC_COL] == tid].copy()
        if sub.empty:
            report_lines.append(f"Topic {tid}: no chunks found.")
            continue

        sub["party_norm"] = sub["party_label"].apply(normalize_text)
        sub = sub[~sub["party_norm"].isin(EXCLUDED_PARTY_LABELS)].copy()
        if sub.empty:
            report_lines.append(f"Topic {tid}: no valid party labels.")
            continue

        if args.parties:
            wanted = {normalize_text(p) for p in args.parties}
            sub = sub[sub["party_norm"].isin(wanted)].copy()
        else:
            counts = sub["party_norm"].value_counts().head(args.top_parties)
            sub = sub[sub["party_norm"].isin(set(counts.index))].copy()

        if sub.empty:
            report_lines.append(f"Topic {tid}: no parties after filtering.")
            continue

        party_display = (
            sub.groupby("party_norm")["party_label"]
            .agg(lambda s: s.value_counts().index[0])
            .to_dict()
        )

        rows = []
        for (party_norm, year), grp in sub.groupby(["party_norm", YEAR_COL]):
            n_chunks = len(grp)
            if n_chunks < args.min_chunks:
                continue
            idx = grp["chunk_index"].values.astype(int)
            centroid = embeddings[idx].mean(axis=0)
            rows.append({
                "topic_id": tid,
                "topic_label": topic_display_name(tid, labels, auto_names),
                "party": party_display.get(party_norm, party_norm),
                "year": year,
                "n_chunks": n_chunks,
                "embedding": centroid,
            })

        if not rows:
            report_lines.append(f"Topic {tid}: no (party, year) cells meet min_chunks.")
            continue

        emb_matrix = np.vstack([r["embedding"] for r in rows])
        coords = project_2d(emb_matrix, args.method, args.random_state)
        for r, (x, y) in zip(rows, coords):
            r["x"] = float(x)
            r["y"] = float(y)

        rows_df = pd.DataFrame(rows)
        rows_df = rows_df.drop(columns=["embedding"])
        rows_df = rows_df.sort_values(["party", "year"])
        out_csv = DATA_DIR / f"topic_{tid}_positions.csv"
        rows_df.to_csv(out_csv, index=False, encoding="utf-8-sig")

        all_rows.append(rows_df)

        title = topic_display_name(tid, labels, auto_names)
        fig, ax = plt.subplots(figsize=(9, 6))
        parties = rows_df["party"].unique().tolist()
        cmap = plt.cm.get_cmap("tab10", max(3, len(parties)))

        for i, party in enumerate(parties):
            g = rows_df[rows_df["party"] == party].sort_values("year")
            ax.plot(g["x"], g["y"], marker="o", label=party, color=cmap(i), alpha=0.8)
            last = g.iloc[-1]
            ax.text(last["x"], last["y"], str(last["year"]), fontsize=8, color=cmap(i))

        ax.set_title(f"{title} — positioning by party over time")
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=8)
        plt.tight_layout()
        fig_path = FIG_DIR / f"topic_{tid}_trajectory.png"
        plt.savefig(fig_path, dpi=160)
        plt.close()

        report_lines.append(
            f"Topic {tid}: {len(rows_df)} points, {len(parties)} parties, saved {out_csv.name} and {fig_path.name}."
        )

    if all_rows:
        all_df = pd.concat(all_rows, ignore_index=True)
        all_df.to_csv(DATA_DIR / "positions_all_topics.csv", index=False, encoding="utf-8-sig")

    report_path = REPORTS_DIR / "topic_positioning_info.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
        f.write("\n")

    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
