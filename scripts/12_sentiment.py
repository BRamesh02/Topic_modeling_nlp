"""
Step 12 — Sentiment by party_family x topic on shared topics.

Score each chunk with cmarkea/distilcamembert-base-sentiment, map the 5-star
output to a continuous score in [-1, 1] via (E[stars] - 3) / 2, then aggregate
per family x topic. Topics shared by at least MIN_PARTIES families with at
least MIN_CHUNKS_PER_CELL chunks each are reported with the range of family
means (= polarisation indicator) and qualitative extracts.

Sentiment scores are cached in chunks_with_sentiment.csv. Delete the file
to rescore from scratch.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = PROJECT_ROOT / "outputs"

CHUNKS_PATH = OUTPUTS / "07_bertopic" / "chunks_with_topics.csv"
PARTY_FAMILY_PATH = OUTPUTS / "09_doc_topic_vectors" / "doc_party_family.csv"
TOPIC_INFO_PATH = OUTPUTS / "07_bertopic" / "topic_info.csv"
TOPIC_LABELS_PATH = OUTPUTS / "07_bertopic" / "topic_labels.csv"

STEP_DIR = OUTPUTS / "12_sentiment"
FIG_DIR = STEP_DIR / "figures"
REPORTS_DIR = STEP_DIR / "reports"
STEP_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

CHUNKS_SENT_PATH = STEP_DIR / "chunks_with_sentiment.csv"
SENT_TABLE_PATH = STEP_DIR / "sentiment_by_party_topic.csv"
SHARED_TOPICS_PATH = STEP_DIR / "shared_topics_polarization.csv"
EXTRACTS_PATH = REPORTS_DIR / "qualitative_extracts.txt"
INFO_PATH = REPORTS_DIR / "sentiment_info.txt"
HEATMAP_PATH = FIG_DIR / "sentiment_heatmap.png"
BOX_PATH = FIG_DIR / "polarized_topics_box.png"

DOC_ID_COL = "doc_id"
TEXT_COL = "chunk_text"
TOPIC_COL = "topic"
CHUNK_ID_COL = "chunk_id"

EXCLUDED_FAMILIES = {"unclassified", "other"}
BOILERPLATE_TOPICS = {5, 15}

MODEL = "cmarkea/distilcamembert-base-sentiment"
BATCH_SIZE = 32
MAX_LENGTH = 256

MIN_CHUNKS_PER_CELL = 10
MIN_PARTIES = 3
TOP_K_POLARIZED = 8
EXTRACTS_PER_CELL = 3


def score_chunks(texts):
    import torch
    from transformers import (
        pipeline,
        CamembertTokenizer,
        CamembertForSequenceClassification,
    )

    if torch.cuda.is_available():
        device = 0
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = -1
    print(f"Loading sentiment model: {MODEL} | device={device}")

    tokenizer = CamembertTokenizer.from_pretrained(MODEL)
    model = CamembertForSequenceClassification.from_pretrained(MODEL)
    pipe = pipeline(
        task="text-classification",
        model=model, tokenizer=tokenizer,
        top_k=None, truncation=True, max_length=MAX_LENGTH, device=device,
    )

    scores = np.zeros(len(texts), dtype=float)
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start:start + BATCH_SIZE]
        results = pipe(batch)

        for i, res in enumerate(results):
            expected = total = 0.0
            for r in res:
                digit = next((int(t) for t in r["label"].lower().replace("_", " ").split() if t.isdigit()), None)
                if digit is None:
                    continue
                if digit == 0:
                    digit = 1
                expected += digit * r["score"]
                total += r["score"]
            scores[start + i] = (expected / total - 3.0) / 2.0 if total > 0 else 0.0

        if start % (BATCH_SIZE * 20) == 0:
            print(f"  scored {min(start + BATCH_SIZE, len(texts))} / {len(texts)}")
    return scores


def load_topic_labels():
    labels = {}
    if TOPIC_INFO_PATH.exists():
        info = pd.read_csv(TOPIC_INFO_PATH)
        labels.update(dict(zip(info["Topic"].astype(int), info["Name"].astype(str))))
    if TOPIC_LABELS_PATH.exists():
        manual = pd.read_csv(TOPIC_LABELS_PATH)
        manual = manual[manual["label"].notna()]
        labels.update(dict(zip(manual["topic_id"].astype(int), manual["label"].astype(str))))
    return labels


def topic_display_name(tid, labels, max_len=35):
    label = labels.get(int(tid), "")
    if not label:
        return f"Topic {tid}"
    return label if len(label) <= max_len else label[: max_len - 1] + "…"


def truncate(text, n=220):
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text).strip()[:n] + "..."


def main():
    print("Loading chunks...")
    chunks = pd.read_csv(CHUNKS_PATH)
    chunks = chunks[chunks[TOPIC_COL] != -1].copy()
    chunks[DOC_ID_COL] = chunks[DOC_ID_COL].astype(str)
    chunks[CHUNK_ID_COL] = chunks[CHUNK_ID_COL].astype(str)

    print("Loading party families...")
    families = pd.read_csv(PARTY_FAMILY_PATH)
    families[DOC_ID_COL] = families[DOC_ID_COL].astype(str)
    chunks = chunks.merge(families[[DOC_ID_COL, "party_family"]], on=DOC_ID_COL, how="left")
    chunks["party_family"] = chunks["party_family"].fillna("unclassified")
    chunks = chunks[~chunks["party_family"].isin(EXCLUDED_FAMILIES)].copy()
    print(f"Chunks after family filter: {len(chunks)}")

    topic_labels = load_topic_labels()

    if CHUNKS_SENT_PATH.exists():
        print(f"Loading cached sentiment: {CHUNKS_SENT_PATH}")
        cached = pd.read_csv(CHUNKS_SENT_PATH)
        cached[CHUNK_ID_COL] = cached[CHUNK_ID_COL].astype(str)
        chunks = chunks.merge(cached[[CHUNK_ID_COL, "sentiment"]], on=CHUNK_ID_COL, how="left")

        missing = chunks["sentiment"].isna()
        if missing.any():
            print(f"Rescoring {missing.sum()} missing chunks...")
            chunks.loc[missing, "sentiment"] = score_chunks(
                chunks.loc[missing, TEXT_COL].fillna("").astype(str).tolist()
            )
    else:
        print("Scoring all chunks...")
        chunks["sentiment"] = score_chunks(chunks[TEXT_COL].fillna("").astype(str).tolist())

    chunks[[CHUNK_ID_COL, "sentiment"]].to_csv(CHUNKS_SENT_PATH, index=False, encoding="utf-8-sig")
    print(f"Saved sentiment cache: {CHUNKS_SENT_PATH}")

    # Per (family, topic) cell
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
    grouped = grouped[grouped["n_chunks"] >= MIN_CHUNKS_PER_CELL].copy()
    grouped["topic_label"] = grouped[TOPIC_COL].map(topic_labels).fillna("")
    grouped.to_csv(SENT_TABLE_PATH, index=False, encoding="utf-8-sig")
    print(f"Saved per-cell sentiment table: {SENT_TABLE_PATH} ({len(grouped)} cells)")

    # Shared topics polarisation
    shared_rows = []
    for tid, sub in grouped.groupby(TOPIC_COL):
        if sub["party_family"].nunique() < MIN_PARTIES:
            continue
        means = sub.set_index("party_family")["mean_sentiment"]
        max_party = means.idxmax()
        min_party = means.idxmin()
        shared_rows.append({
            "topic": int(tid),
            "topic_label": topic_labels.get(int(tid), ""),
            "n_parties": int(sub["party_family"].nunique()),
            "n_chunks_total": int(sub["n_chunks"].sum()),
            "max_party": max_party,
            "max_mean": float(means[max_party]),
            "min_party": min_party,
            "min_mean": float(means[min_party]),
            "range": float(means.max() - means.min()),
            "std": float(means.std()),
        })
    shared = pd.DataFrame(shared_rows).sort_values("range", ascending=False)
    shared.to_csv(SHARED_TOPICS_PATH, index=False, encoding="utf-8-sig")
    print(f"Saved shared topics polarization: {SHARED_TOPICS_PATH} ({len(shared)} shared topics)")

    # Qualitative extracts
    top_polarized = shared.head(TOP_K_POLARIZED)
    with open(EXTRACTS_PATH, "w", encoding="utf-8") as f:
        f.write("=== Qualitative extracts — top polarised shared topics ===\n\n")
        for _, row in top_polarized.iterrows():
            tid = int(row["topic"])
            label = row["topic_label"] or f"Topic {tid}"
            f.write("=" * 90 + "\n")
            f.write(f"TOPIC {tid}  |  {label}\n")
            f.write(f"  range = {row['range']:.3f}  |  std = {row['std']:.3f}  |  n_parties = {row['n_parties']}\n")
            f.write("=" * 90 + "\n\n")

            f.write(f">> {row['max_party']:>22}  (mean sentiment = {row['max_mean']:+.3f})\n")
            sub_max = chunks[(chunks[TOPIC_COL] == tid) & (chunks["party_family"] == row["max_party"])]
            for _, c in sub_max.nlargest(EXTRACTS_PER_CELL, "sentiment").iterrows():
                f.write(f"   [s={c['sentiment']:+.3f}] {truncate(c[TEXT_COL])}\n")
            f.write("\n")

            f.write(f">> {row['min_party']:>22}  (mean sentiment = {row['min_mean']:+.3f})\n")
            sub_min = chunks[(chunks[TOPIC_COL] == tid) & (chunks["party_family"] == row["min_party"])]
            for _, c in sub_min.nsmallest(EXTRACTS_PER_CELL, "sentiment").iterrows():
                f.write(f"   [s={c['sentiment']:+.3f}] {truncate(c[TEXT_COL])}\n")
            f.write("\n\n")
    print(f"Saved qualitative extracts: {EXTRACTS_PATH}")

    corpus_mean = float(chunks["sentiment"].mean())

    # Heatmap
    topic_range = grouped.groupby(TOPIC_COL)["mean_sentiment"].agg(lambda s: s.max() - s.min())
    topic_nfam = grouped.groupby(TOPIC_COL)["party_family"].nunique()
    eligible = sorted(
        topic_range.index[
            (topic_nfam >= 5)
            & (topic_range >= 0.18)
            & (~topic_range.index.isin(BOILERPLATE_TOPICS))
        ],
        key=lambda t: -topic_range[t],
    )

    if eligible:
        sub = grouped[grouped[TOPIC_COL].isin(eligible)]
        pivot = sub.pivot(index="party_family", columns=TOPIC_COL, values="mean_sentiment").loc[:, eligible]

        vmin = min(float(np.nanmin(pivot.values)), corpus_mean - 0.05)
        vmax = max(float(np.nanmax(pivot.values)), corpus_mean + 0.05)
        norm = TwoSlopeNorm(vcenter=corpus_mean, vmin=vmin, vmax=vmax)
        xtick_labels = [topic_display_name(t, topic_labels) for t in pivot.columns]

        fig, ax = plt.subplots(figsize=(max(14, 0.6 * pivot.shape[1]), max(6, 0.85 * pivot.shape[0])))
        im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", norm=norm)
        ax.set_xticks(np.arange(pivot.shape[1]))
        ax.set_xticklabels(xtick_labels, rotation=60, ha="right", fontsize=11)
        ax.set_yticks(np.arange(pivot.shape[0]))
        ax.set_yticklabels(pivot.index, fontsize=12)
        ax.set_xlabel("Topic", fontsize=12)
        ax.set_title(
            f"Mean sentiment by party_family × topic\n"
            f"(centered on corpus mean = {corpus_mean:+.2f}; "
            f"green = relatively positive, red = relatively critical)",
            fontsize=13,
        )
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Mean sentiment", fontsize=12)
        cbar.ax.tick_params(labelsize=10)
        plt.tight_layout()
        plt.savefig(HEATMAP_PATH, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved heatmap: {HEATMAP_PATH}")

    # Boxplots
    box_topics = (
        grouped[(~grouped[TOPIC_COL].isin(BOILERPLATE_TOPICS)) & (grouped["n_chunks"] >= 10)]
        .groupby(TOPIC_COL)
        .agg(
            n_parties=("party_family", "nunique"),
            n_chunks_total=("n_chunks", "sum"),
            min_mean=("mean_sentiment", "min"),
            max_mean=("mean_sentiment", "max"),
        )
        .reset_index()
    )
    box_topics["range"] = box_topics["max_mean"] - box_topics["min_mean"]
    box_topics = box_topics[(box_topics["n_parties"] >= 5) & (box_topics["range"] >= 0.18)]
    box_topics = box_topics.sort_values("range", ascending=False).head(3)

    if not box_topics.empty:
        n = len(box_topics)
        fig, axes = plt.subplots(n, 1, figsize=(13, 3.2 * n), squeeze=False)
        for ax_row, (_, r) in zip(axes, box_topics.iterrows()):
            ax = ax_row[0]
            tid = int(r[TOPIC_COL])
            label = topic_labels.get(tid, f"Topic {tid}")
            sub_chunks = chunks[chunks[TOPIC_COL] == tid]

            family_counts = sub_chunks["party_family"].value_counts()
            parties = family_counts[family_counts >= 10].index.tolist()
            party_means = {p: float(sub_chunks.loc[sub_chunks["party_family"] == p, "sentiment"].mean())
                           for p in parties}
            parties = sorted(parties, key=lambda p: party_means[p])
            data = [sub_chunks.loc[sub_chunks["party_family"] == p, "sentiment"].values for p in parties]

            ax.boxplot(data, tick_labels=parties, vert=True, showfliers=False, patch_artist=True,
                       boxprops=dict(facecolor="#a6c8ed", edgecolor="#234a78"),
                       medianprops=dict(color="#c44e52", linewidth=2.0))
            ax.set_title(
                f"Topic {tid} — {label}  "
                f"(range = {r['range']:.2f}, n = {int(r['n_chunks_total'])} chunks across "
                f"{int(r['n_parties'])} families)",
                fontsize=12, fontweight="bold",
            )
            ax.set_ylabel("Sentiment", fontsize=11)
            ax.axhline(corpus_mean, color="gray", linewidth=0.8, linestyle="--",
                       alpha=0.7, label=f"corpus mean = {corpus_mean:+.2f}")
            ax.legend(loc="lower right", fontsize=9)
            ax.tick_params(axis="x", labelsize=10, rotation=20)
            ax.tick_params(axis="y", labelsize=10)
            ax.grid(axis="y", alpha=0.3)
            ax.set_ylim(-0.3, 1.05)

        plt.tight_layout()
        plt.savefig(BOX_PATH, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved boxplots: {BOX_PATH}")

    # Summary
    with open(INFO_PATH, "w", encoding="utf-8") as f:
        f.write("=== Sentiment by party_family x topic ===\n\n")
        f.write(f"Model: {MODEL}\n")
        f.write(f"Chunks scored: {len(chunks)}\n")
        f.write(f"Cells (party x topic) kept: {len(grouped)} (≥{MIN_CHUNKS_PER_CELL} chunks each)\n")
        f.write(f"Shared topics: {len(shared)} (≥{MIN_PARTIES} parties)\n\n")

        f.write("=== Sentiment distribution overview ===\n")
        f.write(f"  global mean:   {corpus_mean:+.3f}\n")
        f.write(f"  global std:    {chunks['sentiment'].std():.3f}\n")
        f.write(f"  fraction >0:   {(chunks['sentiment'] > 0).mean():.1%}\n")
        f.write(f"  fraction <0:   {(chunks['sentiment'] < 0).mean():.1%}\n\n")

        f.write("=== Top polarised shared topics ===\n\n")
        cols = ["topic", "n_parties", "n_chunks_total", "min_party", "min_mean", "max_party", "max_mean", "range"]
        f.write(shared[cols].head(15).to_string(index=False))
        f.write("\n\n")

        f.write("=== Mean sentiment by party (overall) ===\n")
        party_means = chunks.groupby("party_family")["sentiment"].agg(["mean", "std", "count"]).sort_values("mean")
        f.write(party_means.to_string())
        f.write("\n")

    print(f"Saved summary: {INFO_PATH}")
    print("Done.")


if __name__ == "__main__":
    main()
