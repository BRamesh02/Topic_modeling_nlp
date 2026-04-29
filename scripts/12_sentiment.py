"""
Step 14 — Sentiment by Party x Topic (Brian's H5)

Tests Brian's specific question:
  "When two parties discuss the same topic, do they present it with the same tone?"

Method:
1. Score each chunk with cmarkea/distilcamembert-base-sentiment.
   5-star labels are mapped to a continuous score in [-1, 1] via the expected
   rating: score = (E[stars] - 3) / 2.
2. Aggregate per (party_family, topic) and per (party_family, topic, year).
3. Filter SHARED topics: keep only topics where ≥MIN_PARTIES parties each have
   ≥MIN_CHUNKS scored chunks. This is the prerequisite for "two parties
   discussing the same topic".
4. Polarization metrics per shared topic:
   - range  = max(mean_sentiment) - min(mean_sentiment) across parties
   - std    = standard deviation of party means
   - max(|delta|) and which two parties produce it
5. Qualitative output: for the top-K most polarized shared topics, extract the
   3 most positive chunks (from the highest-sentiment party) and the 3 most
   negative (from the lowest-sentiment party). This material goes into the
   report to show concrete tonal differences.
6. Bonus: sentiment per (party, topic, year) — bridge with Clément's temporal
   analysis. Shows whether a party's tone on a topic shifts across elections.

Inputs:
  outputs/07_bertopic/chunks_with_topics.csv
  outputs/07_bertopic/topic_info.csv             (optional, for topic labels)
  outputs/09_doc_topic_vectors/doc_party_family.csv

Outputs in outputs/12_sentiment/:
  chunks_with_sentiment.csv                      (score cache)
  sentiment_by_party_topic.csv
  sentiment_by_party_topic_year.csv
  shared_topics_polarization.csv
  reports/qualitative_extracts.txt
  reports/sentiment_info.txt
  figures/sentiment_heatmap.png
  figures/polarized_topics_box.png
"""

from __future__ import annotations

from pathlib import Path
import argparse
import ast
import re
import textwrap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = PROJECT_ROOT / "outputs"

CHUNKS_PATH = OUTPUTS / "07_bertopic" / "chunks_with_topics.csv"
PARTY_FAMILY_PATH = OUTPUTS / "09_doc_topic_vectors" / "doc_party_family.csv"
TOPIC_INFO_PATH = OUTPUTS / "07_bertopic" / "topic_info.csv"

STEP_DIR = OUTPUTS / "12_sentiment"
FIG_DIR = STEP_DIR / "figures"
REPORTS_DIR = STEP_DIR / "reports"
STEP_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

CHUNKS_SENT_PATH = STEP_DIR / "chunks_with_sentiment.csv"
SENT_TABLE_PATH = STEP_DIR / "sentiment_by_party_topic.csv"
SENT_YEAR_PATH = STEP_DIR / "sentiment_by_party_topic_year.csv"
SHARED_TOPICS_PATH = STEP_DIR / "shared_topics_polarization.csv"
EXTRACTS_PATH = REPORTS_DIR / "qualitative_extracts.txt"
INFO_PATH = REPORTS_DIR / "sentiment_info.txt"
HEATMAP_PATH = FIG_DIR / "sentiment_heatmap.png"
BOX_PATH = FIG_DIR / "polarized_topics_box.png"


DOC_ID_COL = "doc_id"
TEXT_COL = "chunk_text"
TOPIC_COL = "topic"
CHUNK_ID_COL = "chunk_id"
YEAR_COL = "year"

EXCLUDED_FAMILIES = {"unclassified", "other"}

DEFAULT_MODEL = "cmarkea/distilcamembert-base-sentiment"
DEFAULT_BATCH_SIZE = 32
DEFAULT_MAX_LENGTH = 256

# Cell filter (party x topic): keep only cells with ≥this many chunks
DEFAULT_MIN_CHUNKS_PER_CELL = 10
# Shared-topic filter: a topic is "shared" if ≥this many parties pass MIN_CHUNKS_PER_CELL
DEFAULT_MIN_PARTIES = 3
# Number of polarized topics to drill into qualitatively
DEFAULT_TOP_K_POLARIZED = 8
# Number of extracts per (party, topic) in qualitative output
DEFAULT_EXTRACTS_PER_CELL = 3


def score_chunks(
    texts: list[str],
    model_name: str,
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    """Returns continuous sentiment score in [-1, 1].

    cmarkea/distilcamembert-base-sentiment outputs 5 labels: '1 étoile' .. '5 étoiles'.
    We compute expected rating E[stars] = sum_k k * P(k), then map to [-1, 1] via
    (E - 3) / 2 so 1★ → -1, 3★ → 0, 5★ → +1.
    """
    import torch
    from transformers import (
        pipeline,
        CamembertTokenizer,
        CamembertForSequenceClassification,
    )

    if torch.cuda.is_available():
        device = 0
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = -1

    print(f"Loading sentiment model: {model_name} | device={device}")

    tokenizer = CamembertTokenizer.from_pretrained(model_name)
    model = CamembertForSequenceClassification.from_pretrained(model_name)

    pipe = pipeline(
        task="text-classification",
        model=model,
        tokenizer=tokenizer,
        top_k=None,
        truncation=True,
        max_length=max_length,
        device=device,
    )

    scores = np.zeros(len(texts), dtype=float)

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        results = pipe(batch)

        for i, res in enumerate(results):
            expected = 0.0
            total = 0.0
            for r in res:
                label = r["label"].lower()
                digit = None
                for tok in label.replace("_", " ").split():
                    if tok.isdigit():
                        digit = int(tok)
                        break
                if digit is None:
                    continue
                if digit == 0:
                    digit = 1  # LABEL_0..LABEL_4 → 1..5
                expected += digit * r["score"]
                total += r["score"]

            if total > 0:
                avg_rating = expected / total
                scores[start + i] = (avg_rating - 3.0) / 2.0
            else:
                label_to_p = {r["label"].lower(): r["score"] for r in res}
                scores[start + i] = label_to_p.get("positive", 0.0) - label_to_p.get("negative", 0.0)

        if start % (batch_size * 20) == 0:
            print(f"  scored {min(start + batch_size, len(texts))} / {len(texts)}")

    return scores


def load_topic_labels(path: Path) -> dict[int, str]:
    if not path.exists():
        return {}
    info = pd.read_csv(path)
    if "Topic" not in info.columns or "Name" not in info.columns:
        return {}
    return dict(zip(info["Topic"].astype(int), info["Name"].astype(str)))


def short_label(name: str, max_chars: int = 60) -> str:
    if not isinstance(name, str):
        return ""
    # remove leading "0_", "12_" prefix
    name = re.sub(r"^-?\d+_", "", name)
    name = name.replace("_", " | ")
    return name[:max_chars] + ("..." if len(name) > max_chars else "")


def truncate(text: str, n: int = 220) -> str:
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text).strip()[:n] + "..."


def main() -> None:
    parser = argparse.ArgumentParser(description="Sentiment per party_family x topic.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--min-chunks-per-cell", type=int, default=DEFAULT_MIN_CHUNKS_PER_CELL,
                        help="Minimum chunks for a (party, topic) cell to count.")
    parser.add_argument("--min-parties", type=int, default=DEFAULT_MIN_PARTIES,
                        help="A topic is 'shared' if at least this many parties pass the cell threshold.")
    parser.add_argument("--top-k-polarized", type=int, default=DEFAULT_TOP_K_POLARIZED,
                        help="Number of polarized topics to extract qualitatively.")
    parser.add_argument("--extracts-per-cell", type=int, default=DEFAULT_EXTRACTS_PER_CELL,
                        help="Number of chunks per (party, topic) cell in qualitative extracts.")
    parser.add_argument("--force-rescore", action="store_true")
    args = parser.parse_args()


    print("Loading chunks...")
    chunks = pd.read_csv(CHUNKS_PATH)

    required = {DOC_ID_COL, TEXT_COL, TOPIC_COL, CHUNK_ID_COL}
    missing = required - set(chunks.columns)
    if missing:
        raise ValueError(f"Missing columns in chunks file: {missing}")

    chunks = chunks[chunks[TOPIC_COL] != -1].copy()
    chunks[DOC_ID_COL] = chunks[DOC_ID_COL].astype(str)
    chunks[CHUNK_ID_COL] = chunks[CHUNK_ID_COL].astype(str)

    print("Loading party families...")
    families = pd.read_csv(PARTY_FAMILY_PATH)
    if DOC_ID_COL not in families.columns or "party_family" not in families.columns:
        raise ValueError("doc_party_family.csv missing required columns.")
    families[DOC_ID_COL] = families[DOC_ID_COL].astype(str)

    chunks = chunks.merge(
        families[[DOC_ID_COL, "party_family"]], on=DOC_ID_COL, how="left"
    )
    chunks["party_family"] = chunks["party_family"].fillna("unclassified")
    chunks = chunks[~chunks["party_family"].isin(EXCLUDED_FAMILIES)].copy()
    print(f"Chunks after family filter: {len(chunks)}")

    if chunks.empty:
        raise ValueError("No chunks left after filtering.")

    topic_labels = load_topic_labels(TOPIC_INFO_PATH)


    if CHUNKS_SENT_PATH.exists() and not args.force_rescore:
        print(f"Loading cached sentiment: {CHUNKS_SENT_PATH}")
        cached = pd.read_csv(CHUNKS_SENT_PATH)
        cached[CHUNK_ID_COL] = cached[CHUNK_ID_COL].astype(str)

        chunks = chunks.merge(
            cached[[CHUNK_ID_COL, "sentiment"]], on=CHUNK_ID_COL, how="left"
        )
        missing_mask = chunks["sentiment"].isna()
        if missing_mask.any():
            print(f"Rescoring {missing_mask.sum()} missing chunks...")
            scores = score_chunks(
                chunks.loc[missing_mask, TEXT_COL].fillna("").astype(str).tolist(),
                args.model, args.batch_size, args.max_length,
            )
            chunks.loc[missing_mask, "sentiment"] = scores
    else:
        print("Scoring all chunks...")
        chunks["sentiment"] = score_chunks(
            chunks[TEXT_COL].fillna("").astype(str).tolist(),
            args.model, args.batch_size, args.max_length,
        )

    chunks[[CHUNK_ID_COL, "sentiment"]].to_csv(
        CHUNKS_SENT_PATH, index=False, encoding="utf-8-sig"
    )
    print(f"Saved sentiment cache: {CHUNKS_SENT_PATH}")

    # Aggregate (party, topic)

    grouped = (
        chunks.groupby(["party_family", TOPIC_COL])
        .agg(
            mean_sentiment=("sentiment", "mean"),
            std_sentiment=("sentiment", "std"),
            n_chunks=("sentiment", "size"),
            n_docs=(DOC_ID_COL, "nunique"),
        )
        .reset_index()
    )

    # Filter cells with enough chunks
    grouped = grouped[grouped["n_chunks"] >= args.min_chunks_per_cell].copy()
    grouped["topic_label"] = grouped[TOPIC_COL].map(topic_labels).fillna("")
    grouped.to_csv(SENT_TABLE_PATH, index=False, encoding="utf-8-sig")
    print(f"Saved per-cell sentiment table: {SENT_TABLE_PATH} ({len(grouped)} cells)")


    shared_rows = []
    for tid, sub in grouped.groupby(TOPIC_COL):
        if sub["party_family"].nunique() < args.min_parties:
            continue
        means = sub.set_index("party_family")["mean_sentiment"]
        sentiment_range = float(means.max() - means.min())
        sentiment_std = float(means.std())
        max_party = means.idxmax()
        min_party = means.idxmin()
        n_total_chunks = int(sub["n_chunks"].sum())

        shared_rows.append({
            "topic": int(tid),
            "topic_label": topic_labels.get(int(tid), ""),
            "n_parties": int(sub["party_family"].nunique()),
            "n_chunks_total": n_total_chunks,
            "max_party": max_party,
            "max_mean": float(means[max_party]),
            "min_party": min_party,
            "min_mean": float(means[min_party]),
            "range": sentiment_range,
            "std": sentiment_std,
        })

    shared = pd.DataFrame(shared_rows).sort_values("range", ascending=False)
    shared.to_csv(SHARED_TOPICS_PATH, index=False, encoding="utf-8-sig")
    print(f"Saved shared topics polarization: {SHARED_TOPICS_PATH} ({len(shared)} shared topics)")

    # Sentiment per (party, topic, year)

    if YEAR_COL in chunks.columns:
        grouped_y = (
            chunks.dropna(subset=[YEAR_COL])
            .groupby(["party_family", TOPIC_COL, YEAR_COL])
            .agg(
                mean_sentiment=("sentiment", "mean"),
                n_chunks=("sentiment", "size"),
            )
            .reset_index()
        )
        grouped_y = grouped_y[grouped_y["n_chunks"] >= args.min_chunks_per_cell].copy()
        grouped_y["topic_label"] = grouped_y[TOPIC_COL].map(topic_labels).fillna("")
        grouped_y.to_csv(SENT_YEAR_PATH, index=False, encoding="utf-8-sig")
        print(f"Saved per-year sentiment table: {SENT_YEAR_PATH} ({len(grouped_y)} cells)")

    # Qualitative extracts for top-K polarized topics

    top_polarized = shared.head(args.top_k_polarized)

    with open(EXTRACTS_PATH, "w", encoding="utf-8") as f:
        f.write("=== QUALITATIVE EXTRACTS — TOP POLARIZED SHARED TOPICS ===\n\n")
        f.write(f"For each topic, the {args.extracts_per_cell} most positive chunks from the\n")
        f.write("highest-sentiment party and the most negative chunks from the lowest.\n")
        f.write("Use these as concrete examples in the report.\n\n")

        for _, row in top_polarized.iterrows():
            tid = int(row["topic"])
            label = row["topic_label"] or f"Topic {tid}"
            f.write("=" * 90 + "\n")
            f.write(f"TOPIC {tid}  |  {label}\n")
            f.write(f"  range = {row['range']:.3f}  |  std = {row['std']:.3f}  |  n_parties = {row['n_parties']}\n")
            f.write("=" * 90 + "\n\n")

            # Most positive party
            f.write(f">> {row['max_party']:>22}  (mean sentiment = {row['max_mean']:+.3f})\n")
            sub_max = chunks[
                (chunks[TOPIC_COL] == tid) & (chunks["party_family"] == row["max_party"])
            ].nlargest(args.extracts_per_cell, "sentiment")
            for _, c in sub_max.iterrows():
                f.write(f"   [s={c['sentiment']:+.3f}] {truncate(c[TEXT_COL])}\n")
            f.write("\n")

            # Most negative party
            f.write(f">> {row['min_party']:>22}  (mean sentiment = {row['min_mean']:+.3f})\n")
            sub_min = chunks[
                (chunks[TOPIC_COL] == tid) & (chunks["party_family"] == row["min_party"])
            ].nsmallest(args.extracts_per_cell, "sentiment")
            for _, c in sub_min.iterrows():
                f.write(f"   [s={c['sentiment']:+.3f}] {truncate(c[TEXT_COL])}\n")
            f.write("\n\n")

    print(f"Saved qualitative extracts: {EXTRACTS_PATH}")

    # Heatmap (party x topic, mean sentiment)

    if not grouped.empty:
        # Limit to topics with at least min_parties parties for readability
        eligible_topics = (
            grouped.groupby(TOPIC_COL)["party_family"].nunique()
            .loc[lambda s: s >= args.min_parties]
            .sort_values(ascending=False)
            .head(40)
            .index
        )
        sub = grouped[grouped[TOPIC_COL].isin(eligible_topics)]
        if not sub.empty:
            pivot = sub.pivot(index="party_family", columns=TOPIC_COL, values="mean_sentiment")
            pivot = pivot.loc[:, sorted(pivot.columns)]

            fig, ax = plt.subplots(figsize=(max(10, 0.4 * pivot.shape[1]), max(4, 0.55 * pivot.shape[0])))
            vmax = max(0.3, np.nanmax(np.abs(pivot.values)))
            im = ax.imshow(pivot.values, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            ax.set_xticks(np.arange(pivot.shape[1]))
            ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=8)
            ax.set_yticks(np.arange(pivot.shape[0]))
            ax.set_yticklabels(pivot.index)
            ax.set_xlabel("Topic id")
            ax.set_title("Mean sentiment by party_family x topic (shared topics only)")
            plt.colorbar(im, ax=ax, label="Mean sentiment (-1 = neg, +1 = pos)")
            plt.tight_layout()
            plt.savefig(HEATMAP_PATH, dpi=150)
            plt.close()
            print(f"Saved heatmap: {HEATMAP_PATH}")

    # Boxplot of sentiment per party for top polarized topics

    if not top_polarized.empty:
        fig, axes = plt.subplots(
            len(top_polarized), 1,
            figsize=(10, 1.6 * len(top_polarized)),
            squeeze=False,
        )

        for ax_row, (_, r) in zip(axes, top_polarized.iterrows()):
            ax = ax_row[0]
            tid = int(r["topic"])
            sub = chunks[chunks[TOPIC_COL] == tid]
            parties = sub["party_family"].unique()
            data = [sub.loc[sub["party_family"] == p, "sentiment"].values for p in parties]

            ax.boxplot(data, labels=parties, vert=True, showfliers=False)
            label = r["topic_label"] or f"Topic {tid}"
            ax.set_title(f"Topic {tid}: {short_label(label, 70)}  (range={r['range']:.2f})", fontsize=9)
            ax.set_ylabel("sentiment")
            ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
            ax.tick_params(axis="x", labelsize=8, rotation=20)

        plt.tight_layout()
        plt.savefig(BOX_PATH, dpi=150)
        plt.close()
        print(f"Saved boxplots: {BOX_PATH}")

    # Per-year sentiment evolution (top polarized topics)

    if YEAR_COL in chunks.columns and not top_polarized.empty:
        try:
            grouped_y_full = (
                chunks.dropna(subset=[YEAR_COL])
                .groupby(["party_family", TOPIC_COL, YEAR_COL])
                .agg(mean_sentiment=("sentiment", "mean"), n=("sentiment", "size"))
                .reset_index()
            )
            grouped_y_full = grouped_y_full[grouped_y_full["n"] >= args.min_chunks_per_cell]

            top_topics = top_polarized["topic"].head(min(6, len(top_polarized))).tolist()
            sub = grouped_y_full[grouped_y_full[TOPIC_COL].isin(top_topics)]

            ncols = 2
            nrows = (len(top_topics) + ncols - 1) // ncols
            fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 3.5 * nrows), squeeze=False)
            for idx, tid in enumerate(top_topics):
                ax = axes[idx // ncols][idx % ncols]
                topic_df = sub[sub[TOPIC_COL] == tid]
                for fam, g in topic_df.groupby("party_family"):
                    g = g.sort_values(YEAR_COL)
                    ax.plot(g[YEAR_COL].astype(str), g["mean_sentiment"],
                            marker="o", label=fam, alpha=0.85)
                label = (top_polarized[top_polarized["topic"] == tid]["topic_label"].values[:1] or [""])[0]
                ax.set_title(f"Topic {tid}: {label[:60]}", fontsize=10)
                ax.set_ylabel("Mean sentiment")
                ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
                ax.grid(alpha=0.3)
                ax.tick_params(axis="x", labelsize=9)
            for idx in range(len(top_topics), nrows * ncols):
                axes[idx // ncols][idx % ncols].axis("off")
            handles, labels = axes[0][0].get_legend_handles_labels()
            if handles:
                fig.legend(handles, labels, loc="lower center", ncol=min(7, len(handles)),
                           fontsize=9, bbox_to_anchor=(0.5, -0.02))
            plt.suptitle("Sentiment evolution by year — top polarized shared topics", y=1.0)
            plt.tight_layout()
            plt.savefig(FIG_DIR / "sentiment_by_year.png", dpi=150, bbox_inches="tight")
            plt.close()
            print(f"Saved year-by-year figure: {FIG_DIR / 'sentiment_by_year.png'}")
        except Exception as e:
            print(f"Could not generate sentiment_by_year figure: {e}")


    with open(INFO_PATH, "w", encoding="utf-8") as f:
        f.write("=== Sentiment by Party x Topic — Brian's H5 ===\n\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Chunks scored: {len(chunks)}\n")
        f.write(f"Cells (party x topic) kept: {len(grouped)}\n")
        f.write(f"  threshold: ≥{args.min_chunks_per_cell} chunks per cell\n")
        f.write(f"Shared topics: {len(shared)}\n")
        f.write(f"  threshold: ≥{args.min_parties} parties per topic\n\n")

        f.write("=== Sentiment distribution overview ===\n")
        f.write(f"  global mean:   {chunks['sentiment'].mean():+.3f}\n")
        f.write(f"  global std:    {chunks['sentiment'].std():.3f}\n")
        f.write(f"  fraction >0:   {(chunks['sentiment'] > 0).mean():.1%}\n")
        f.write(f"  fraction <0:   {(chunks['sentiment'] < 0).mean():.1%}\n\n")

        f.write("=== Top polarized shared topics ===\n")
        f.write("(topics where ≥{n} parties differ most in tone)\n\n".format(n=args.min_parties))
        cols = ["topic", "n_parties", "n_chunks_total", "min_party", "min_mean", "max_party", "max_mean", "range"]
        f.write(shared[cols].head(15).to_string(index=False))
        f.write("\n\n")

        f.write("=== Mean sentiment by party (overall) ===\n")
        party_means = (
            chunks.groupby("party_family")["sentiment"]
            .agg(["mean", "std", "count"])
            .sort_values("mean")
        )
        f.write(party_means.to_string())
        f.write("\n\n")

        f.write("=== Files produced ===\n")
        f.write(f"  per-cell:       {SENT_TABLE_PATH.name}\n")
        f.write(f"  per-cell-year:  {SENT_YEAR_PATH.name}\n")
        f.write(f"  shared topics:  {SHARED_TOPICS_PATH.name}\n")
        f.write(f"  extracts:       {EXTRACTS_PATH.name}\n")
        f.write(f"  heatmap:        {HEATMAP_PATH.name}\n")
        f.write(f"  boxplots:       {BOX_PATH.name}\n")

    print(f"Saved summary: {INFO_PATH}")
    print("Done.")


if __name__ == "__main__":
    main()
