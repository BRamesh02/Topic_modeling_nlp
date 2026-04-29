"""
Step 11 — Visualizations (merges old 13 + 15 + 16)

  --viz projection      : 2D projection of documents in topic space (PCA/t-SNE)
                          + facet by year + interactive plotly (was 13)
  --viz sanity          : digest of BERTopic topics: top words + 3 representative
                          chunks per topic + flags (was 15)
  --viz native          : BERTopic native views topics_per_class + topics_over_time
                          (was 16)
  --all                 : run all three

Inputs:
  outputs/07_bertopic/topic_info.csv
  outputs/07_bertopic/models/bertopic_model/
  outputs/07_bertopic/chunks_with_topics.csv
  outputs/09_doc_topic_vectors/doc_topic_vectors_bertopic.csv
  outputs/09_doc_topic_vectors/doc_party_family.csv

Outputs in outputs/11_visualizations/:
  projection/  doc_topic_projection.csv + figures/doc_projection_*.png|.html
  sanity/      topic_flags.csv + reports/topic_digest.txt
  native/      topics_per_class.csv + topics_over_time.csv
               + figures/topics_per_class.html|.png + figures/topics_over_time.html|.png
"""

from __future__ import annotations

from pathlib import Path
import argparse
import ast
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = PROJECT_ROOT / "outputs"

TOPIC_INFO_PATH = OUTPUTS / "07_bertopic" / "topic_info.csv"
MODEL_DIR = OUTPUTS / "07_bertopic" / "models" / "bertopic_model"
CHUNKS_TOPICS_PATH = OUTPUTS / "07_bertopic" / "chunks_with_topics.csv"
DOC_VEC_PATH = OUTPUTS / "09_doc_topic_vectors" / "doc_topic_vectors_bertopic.csv"
PARTY_FAMILY_PATH = OUTPUTS / "09_doc_topic_vectors" / "doc_party_family.csv"

STEP_DIR = OUTPUTS / "11_visualizations"
STEP_DIR.mkdir(parents=True, exist_ok=True)


DOC_ID_COL = "doc_id"
TEXT_COL = "chunk_text"
TOPIC_COL = "topic"
YEAR_COL = "year"
EXCLUDED_FAMILIES = {"unclassified", "other"}

PARTY_COLORS = {
    "radical_left":         "#7a0000",
    "communist_left":       "#c40000",
    "socialist_left":       "#ff6961",
    "ecologist":            "#1ea672",
    "liberal_center_right": "#fbbf24",
    "gaullist_right":       "#1e40af",
    "national_right":       "#000000",
}


def save_plotly_png(fig, png_path: Path, width: int = 1400, height: int = 800):
    """Save plotly figure as PNG only. Requires kaleido."""
    try:
        fig.write_image(png_path, width=width, height=height, scale=2)
    except Exception as e:
        print(f"PNG export failed for {png_path.name}: {e}")
        print("Install kaleido: pip install kaleido")


# Projection (was script 13)

def _safe_perplexity(n: int, desired: int) -> int:
    if n < 3:
        return 2
    return max(2, min(desired, (n - 1) // 3))


def project(vectors: np.ndarray, method: str, random_state: int) -> np.ndarray:
    if method == "pca":
        return PCA(n_components=2, random_state=random_state).fit_transform(vectors)
    if method == "tsne":
        return TSNE(
            n_components=2,
            perplexity=_safe_perplexity(len(vectors), 30),
            learning_rate=200,
            init="pca",
            random_state=random_state,
        ).fit_transform(vectors)
    raise ValueError(f"Unknown method: {method}")


def run_projection(args) -> None:
    out_dir = STEP_DIR / "projection"
    out_figs = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_figs.mkdir(parents=True, exist_ok=True)

    print("\n[projection] Loading doc-level topic vectors...")
    df = pd.read_csv(DOC_VEC_PATH)
    topic_cols = [c for c in df.columns if c.startswith("topic_")]
    if not topic_cols:
        raise ValueError("No topic_* columns.")

    df = df[~df["party_family"].isin(EXCLUDED_FAMILIES)].copy()
    df = df.dropna(subset=topic_cols, how="all")
    df[topic_cols] = df[topic_cols].fillna(0.0)
    if len(df) == 0:
        raise ValueError("No docs left after filtering.")

    print(f"[projection] Projecting {len(df)} docs with {args.method.upper()}...")
    coords = project(df[topic_cols].values.astype(float), args.method, args.random_state)
    df["x"] = coords[:, 0]
    df["y"] = coords[:, 1]
    df.to_csv(out_dir / "doc_topic_projection.csv", index=False, encoding="utf-8-sig")

    plt.figure(figsize=(10, 7))
    for family, sub in df.groupby("party_family"):
        plt.scatter(sub["x"], sub["y"], label=family,
                    color=PARTY_COLORS.get(family, "gray"),
                    alpha=0.6, s=18, edgecolors="none")
    plt.title(f"Documents in topic space ({args.method.upper()})")
    plt.xlabel("Component 1"); plt.ylabel("Component 2")
    plt.legend(loc="best", fontsize=8, markerscale=1.5)
    plt.tight_layout()
    plt.savefig(out_figs / "doc_projection.png", dpi=150)
    plt.close()

    if "year" in df.columns:
        years = sorted(df["year"].dropna().unique())
        n = len(years)
        if n > 0:
            ncols = min(3, n)
            nrows = (n + ncols - 1) // ncols
            fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
            for idx, year in enumerate(years):
                ax = axes[idx // ncols][idx % ncols]
                sub_year = df[df["year"] == year]
                for family, g in sub_year.groupby("party_family"):
                    ax.scatter(g["x"], g["y"], label=family,
                               color=PARTY_COLORS.get(family, "gray"),
                               alpha=0.6, s=18, edgecolors="none")
                ax.set_title(f"{year} (n={len(sub_year)})")
                ax.set_xlabel("Comp 1"); ax.set_ylabel("Comp 2"); ax.grid(alpha=0.3)
            for idx in range(len(years), nrows * ncols):
                axes[idx // ncols][idx % ncols].axis("off")
            handles, labels = axes[0][0].get_legend_handles_labels()
            if handles:
                fig.legend(handles, labels, loc="lower center", ncol=min(7, len(handles)),
                           fontsize=8, bbox_to_anchor=(0.5, -0.02))
            plt.tight_layout()
            plt.savefig(out_figs / "doc_projection_by_year.png", dpi=150, bbox_inches="tight")
            plt.close()

    print(f"[projection] Done → {out_dir}")


# Sanity check (was script 15)

def parse_repr(value):
    if not isinstance(value, str):
        return []
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return []


def truncate(text: str, n: int = 220) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\s+", " ", text).strip()
    return text[:n] + ("..." if len(text) > n else "")


def flag_topic(top_words: list[str], chunks: list[str]) -> list[str]:
    flags = []
    if not top_words:
        flags.append("NO_WORDS")
        return flags
    generic = {"circonscription", "candidat", "élection", "election", "majorité",
               "majorite", "politique", "pays", "français", "francais", "vote",
               "voter", "monsieur", "madame", "union"}
    generic_count = sum(1 for w in top_words[:8] if w.split()[0] in generic)
    if generic_count >= 4:
        flags.append("GENERIC_BOILERPLATE")
    word_tokens = [w.split()[0] for w in top_words[:8]]
    if len(set(word_tokens)) <= 3:
        flags.append("REPETITIVE_WORDS")
    if chunks:
        prefixes = [c[:60].lower() for c in chunks if isinstance(c, str)]
        if len(set(prefixes)) == 1 and len(prefixes) > 1:
            flags.append("DUPLICATE_CHUNKS")
    return flags


def run_sanity(args) -> None:
    out_dir = STEP_DIR / "sanity"
    out_reports = out_dir / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_reports.mkdir(parents=True, exist_ok=True)

    print(f"\n[sanity] Loading {TOPIC_INFO_PATH}...")
    info = pd.read_csv(TOPIC_INFO_PATH)
    info["words"] = info["Representation"].apply(parse_repr)
    info["docs"] = info["Representative_Docs"].apply(parse_repr)
    info_sorted = info.sort_values("Count", ascending=False)

    flag_rows = []
    digest_path = out_reports / "topic_digest.txt"
    with open(digest_path, "w", encoding="utf-8") as f:
        f.write("=== TOPIC SANITY DIGEST ===\n")
        f.write(f"Source: {TOPIC_INFO_PATH}\n")
        f.write(f"Topics: {len(info)}\n\n")

        for _, row in info_sorted.iterrows():
            tid = row["Topic"]; count = row["Count"]; name = row.get("Name", "")
            words = row["words"][:8]; docs = row["docs"][:3]
            flags = flag_topic(words, docs)
            flag_str = ", ".join(flags) if flags else "OK"
            flag_rows.append({"topic": tid, "count": count, "name": name, "flags": flag_str})

            f.write(f"\n{'=' * 78}\n")
            f.write(f"Topic {tid:>4}  |  count={count:>6}  |  flags: {flag_str}\n")
            f.write(f"{'=' * 78}\n")
            f.write(f"Name:  {name}\n")
            f.write(f"Words: {' | '.join(str(w) for w in words)}\n\n")
            for i, doc in enumerate(docs, 1):
                f.write(f"  [Rep doc {i}] {truncate(doc)}\n\n")

    flags_df = pd.DataFrame(flag_rows)
    flags_df.to_csv(out_dir / "topic_flags.csv", index=False, encoding="utf-8-sig")

    n_ok = (flags_df["flags"] == "OK").sum()
    print(f"[sanity] Done → {out_dir}  |  OK: {n_ok}  flagged: {len(flags_df) - n_ok}")


# BERTopic native views (was script 16)

def run_native(args) -> None:
    out_dir = STEP_DIR / "native"
    out_figs = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_figs.mkdir(parents=True, exist_ok=True)

    print("\n[native] Loading BERTopic model...")
    from bertopic import BERTopic
    topic_model = BERTopic.load(str(MODEL_DIR))

    print("[native] Loading chunks...")
    # IMPORTANT: do NOT filter rows here. BERTopic's topics_per_class and
    # topics_over_time use self.topics_ internally, which has the same length
    # as the chunks the model was fit on. Filtering before would create a
    # length mismatch ("All arrays must be of the same length"). We filter
    # the OUTPUT instead.
    chunks = pd.read_csv(CHUNKS_TOPICS_PATH)
    chunks[DOC_ID_COL] = chunks[DOC_ID_COL].astype(str)

    families = pd.read_csv(PARTY_FAMILY_PATH)
    families[DOC_ID_COL] = families[DOC_ID_COL].astype(str)
    chunks = chunks.merge(families[[DOC_ID_COL, "party_family"]], on=DOC_ID_COL, how="left")
    chunks["party_family"] = chunks["party_family"].fillna("unclassified")
    print(f"[native] Chunks (full, no filter): {len(chunks)}")

    docs = chunks[TEXT_COL].fillna("").astype(str).tolist()

    print("[native] topics_per_class...")
    classes = chunks["party_family"].tolist()
    tpc = topic_model.topics_per_class(docs, classes=classes, global_tuning=args.global_tuning)

    # Filter output: drop noise topic and excluded families
    tpc = tpc[tpc["Topic"] != -1]
    tpc = tpc[~tpc["Class"].isin(EXCLUDED_FAMILIES)]
    tpc.to_csv(out_dir / "topics_per_class.csv", index=False, encoding="utf-8-sig")

    try:
        fig = topic_model.visualize_topics_per_class(tpc, top_n_topics=args.top_n_topics)
        save_plotly_png(fig, out_figs / "topics_per_class.png", width=1400, height=900)
    except Exception as e:
        print(f"[native] Could not save topics_per_class figure: {e}")

    if YEAR_COL in chunks.columns:
        print("[native] topics_over_time...")
        # Same constraint: full length needed. Fill missing years with placeholder.
        years_full = chunks[YEAR_COL].copy()
        years_filled = years_full.fillna(-1).astype(int).astype(str).tolist()
        n_unique = sum(1 for y in set(years_filled) if y != "-1")
        nr_bins = min(args.nr_bins, n_unique)

        tot = topic_model.topics_over_time(
            docs, timestamps=years_filled,
            global_tuning=args.global_tuning, nr_bins=nr_bins,
        )
        # Filter output: drop noise topic and the placeholder year
        tot = tot[tot["Topic"] != -1]
        tot = tot[~tot["Timestamp"].astype(str).str.contains("-1")]
        tot.to_csv(out_dir / "topics_over_time.csv", index=False, encoding="utf-8-sig")

        try:
            fig = topic_model.visualize_topics_over_time(tot, top_n_topics=args.top_n_topics)
            save_plotly_png(fig, out_figs / "topics_over_time.png", width=1400, height=700)
        except Exception as e:
            print(f"[native] Could not save topics_over_time figure: {e}")

    print(f"[native] Done → {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Visualizations: projection / sanity / native.")
    parser.add_argument("--viz", type=str, choices=["projection", "sanity", "native"], default=None)
    parser.add_argument("--all", action="store_true")
    # projection params
    parser.add_argument("--method", choices=["pca", "tsne"], default="pca")
    parser.add_argument("--random-state", type=int, default=42)
    # native params
    parser.add_argument("--top-n-topics", type=int, default=15)
    parser.add_argument("--global-tuning", action="store_true")
    parser.add_argument("--nr-bins", type=int, default=5)

    args = parser.parse_args()

    if not args.all and args.viz is None:
        parser.error("Specify --viz {projection,sanity,native} or --all")

    if args.all or args.viz == "projection":
        run_projection(args)
    if args.all or args.viz == "sanity":
        run_sanity(args)
    if args.all or args.viz == "native":
        run_native(args)

    print("\nDone.")


if __name__ == "__main__":
    main()
