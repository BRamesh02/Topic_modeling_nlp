"""
Step 9 — Aggregate chunk-level BERTopic topics into document-level topic
vectors and attach a party_family label per document. Outlier topic (-1)
is dropped before aggregation.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _family_mapping import assign_party_family


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = PROJECT_ROOT / "outputs"

BERTOPIC_DIR = OUTPUTS / "07_bertopic"
STEP_DIR = OUTPUTS / "09_doc_topic_vectors"
REPORTS_DIR = STEP_DIR / "reports"
FIG_DIR = STEP_DIR / "figures"

STEP_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

BERTOPIC_CHUNKS_PATH = BERTOPIC_DIR / "chunks_with_topics.csv"
BERTOPIC_VEC_PATH = STEP_DIR / "doc_topic_vectors_bertopic.csv"
PARTY_FAMILY_PATH = STEP_DIR / "doc_party_family.csv"
INFO_PATH = REPORTS_DIR / "doc_topic_vectors_info.txt"


DOC_ID_COL = "doc_id"
TOPIC_COL = "topic"

DOC_META_COLS = [
    "year", "date", "contexte-tour",
    "titulaire-liste", "titulaire-soutien",
    "titulaire-sexe", "titulaire-prenom", "titulaire-nom",
]


def main():
    print("\nLoading BERTopic chunks...")
    df = pd.read_csv(BERTOPIC_CHUNKS_PATH)
    print(f"Chunks loaded: {len(df)}")

    # Build doc-level topic distribution (drop outliers)
    work = df[[DOC_ID_COL, TOPIC_COL]].copy()
    work = work[work[TOPIC_COL] != -1]
    work["weight"] = 1.0
    pivot = work.pivot_table(
        index=DOC_ID_COL, columns=TOPIC_COL, values="weight",
        aggfunc="sum", fill_value=0.0,
    ).sort_index(axis=1)
    pivot = pivot.div(pivot.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    pivot.columns = [f"topic_{int(c)}" for c in pivot.columns]

    # Doc-level metadata + family mapping
    meta_cols = [c for c in DOC_META_COLS if c in df.columns]
    meta = df[[DOC_ID_COL] + meta_cols].groupby(DOC_ID_COL, as_index=False).first()
    meta["party_family"] = meta.apply(assign_party_family, axis=1)

    # Merge metadata + topic vectors
    out = meta.merge(pivot.reset_index(), on=DOC_ID_COL, how="left").fillna(0.0)
    topic_cols = [c for c in out.columns if c.startswith("topic_")]
    out["dominant_topic"] = out[topic_cols].idxmax(axis=1).str.replace("topic_", "").astype(int)
    out.to_csv(BERTOPIC_VEC_PATH, index=False, encoding="utf-8-sig")

    family_cols = [DOC_ID_COL, "party_family"] + meta_cols
    meta[family_cols].to_csv(PARTY_FAMILY_PATH, index=False, encoding="utf-8-sig")

    with open(INFO_PATH, "w", encoding="utf-8") as f:
        f.write("=== Doc-level Topic Vectors ===\n\n")
        f.write(f"Docs: {len(out)}, topics: {len(topic_cols)}\n\n")
        f.write("=== Party family counts ===\n")
        f.write(out["party_family"].value_counts().to_string())
        f.write("\n\n=== Docs by year ===\n")
        f.write(out["year"].value_counts().sort_index().to_string())
        f.write("\n\n=== Dominant topics ===\n")
        f.write(out["dominant_topic"].value_counts().head(20).to_string())
        f.write("\n")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    counts = out["party_family"].value_counts()
    counts.plot(kind="barh", ax=axes[0], color="#4a7ab5")
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Number of documents")
    axes[0].set_title("Documents per party family")
    for i, v in enumerate(counts.values):
        axes[0].text(v + max(counts.values) * 0.01, i, str(v), va="center", fontsize=9)

    ct = pd.crosstab(out["year"], out["party_family"])
    valid = [c for c in ct.columns if c not in ("unclassified", "other")]
    ct[valid].plot(kind="bar", stacked=True, ax=axes[1], colormap="tab10")
    axes[1].set_xlabel("Year")
    axes[1].set_ylabel("Number of documents")
    axes[1].set_title("Party family distribution by year")
    axes[1].legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)
    axes[1].tick_params(axis="x", rotation=0)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "party_family_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("\nDone.")
    print(f"Saved doc vectors: {BERTOPIC_VEC_PATH}")
    print(f"Saved party_family: {PARTY_FAMILY_PATH}")
    print(f"Saved figure: {FIG_DIR / 'party_family_distribution.png'}")


if __name__ == "__main__":
    main()
