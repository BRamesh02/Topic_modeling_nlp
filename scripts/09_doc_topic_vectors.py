"""
Step 9 — Document-level Topic Vectors + Party Family Assignment

Consolidates BERTopic and LDA chunk-level outputs into doc-level topic
distributions, and assigns each document to a party_family. This is the single
source of truth used by steps 10-13.

Inputs:
data/chunks_with_topics.csv          (BERTopic, column 'topic')
data/chunks_with_topics_lda.csv      (LDA, columns 'topic_lda', 'topic_lda_score')

Outputs:
outputs/doc_topic_vectors_bertopic.csv
outputs/doc_topic_vectors_lda.csv
outputs/doc_party_family.csv
outputs/doc_topic_vectors_info.txt
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List
import argparse
import re

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = PROJECT_ROOT / "outputs"

BERTOPIC_DIR = OUTPUTS / "07_bertopic"
LDA_DIR = OUTPUTS / "08_lda"
STEP_DIR = OUTPUTS / "09_doc_topic_vectors"
REPORTS_DIR = STEP_DIR / "reports"
FIG_DIR = STEP_DIR / "figures"

STEP_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

BERTOPIC_CHUNKS_PATH = BERTOPIC_DIR / "chunks_with_topics.csv"
LDA_CHUNKS_PATH = LDA_DIR / "chunks_with_topics_lda.csv"

BERTOPIC_VEC_PATH = STEP_DIR / "doc_topic_vectors_bertopic.csv"
LDA_VEC_PATH = STEP_DIR / "doc_topic_vectors_lda.csv"
PARTY_FAMILY_PATH = STEP_DIR / "doc_party_family.csv"
INFO_PATH = REPORTS_DIR / "doc_topic_vectors_info.txt"


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

AFFILIATION_FIELDS = ["titulaire-soutien", "titulaire-liste"]


def _compact(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).lower()


def assign_party_family(row: pd.Series) -> str:
    """Map raw party labels (titulaire-soutien / titulaire-liste) to coarse families.
    Order matters: more specific patterns checked first (e.g. 'parti socialiste unifié'
    -> radical_left, before generic 'socialiste' -> socialist_left).
    """
    # Drop "non mentionné" from each field BEFORE concatenation, so a doc with
    # one informative field and one missing one is still classified.
    parts = []
    for col in AFFILIATION_FIELDS:
        val = _compact(row.get(col, ""))
        if val and "non mentionné" not in val:
            parts.append(val)
    labels = " ".join(parts)

    if not labels.strip():
        return "unclassified"

    # ===== National / far right =====
    if "front national" in labels:
        return "national_right"
    if "parti des forces nouvelles" in labels:
        return "national_right"

    # ===== Radical left (must be checked BEFORE 'communiste' and 'socialiste') =====
    if "lutte ouvrière" in labels or "lutte ouvriere" in labels:
        return "radical_left"
    if "ligue communiste" in labels:  # LC (1973) + LCR (1978)
        return "radical_left"
    if "parti socialiste unifié" in labels or "parti socialiste unifie" in labels:  # PSU
        return "radical_left"
    if "parti ouvrier européen" in labels or "parti ouvrier europeen" in labels:  # POE 1988
        return "radical_left"
    if "comités juquin" in labels or "comites juquin" in labels:  # 1988
        return "radical_left"
    if "marxistes-léninistes" in labels or "marxistes-leninistes" in labels:
        return "radical_left"

    # ===== Communist (PCF) — after 'ligue communiste' =====
    if "communiste" in labels:
        return "communist_left"

    # ===== Ecologist (incl. 1993 specific lists) =====
    if "écolog" in labels or "ecolog" in labels:
        return "ecologist"
    if "verts" in labels:
        return "ecologist"
    if "nature et animaux" in labels:
        return "ecologist"
    if "défense des animaux" in labels or "defense des animaux" in labels:
        return "ecologist"

    # ===== Socialist (PS, MRG, PSD) — must be after PSU =====
    if "parti socialiste démocrate" in labels or "parti socialiste democrate" in labels:
        return "socialist_left"
    if "socialiste" in labels:
        return "socialist_left"
    if "radicaux de gauche" in labels or "radical de gauche" in labels:
        return "socialist_left"
    if "mouvement des citoyens" in labels:  # MDC Chevènement (1993+)
        return "socialist_left"

    # ===== Gaullist right (UDR / RPR / URP) =====
    if "rassemblement pour la république" in labels or "rassemblement pour la republique" in labels:
        return "gaullist_right"
    if re.search(r"\brpr\b", labels):
        return "gaullist_right"
    if "union des démocrates pour la république" in labels or "union des democrates pour la republique" in labels:
        return "gaullist_right"
    if re.search(r"\budr\b", labels):
        return "gaullist_right"
    if "union des républicains de progrès" in labels or "union des republicains de progres" in labels:
        return "gaullist_right"
    if re.search(r"\burp\b", labels):
        return "gaullist_right"
    if "gaulliste" in labels or "gaullistes" in labels:  # gaulliste, Union/Fédération des gaullistes
        return "gaullist_right"

    # ===== Liberal / center-right (UDF, MR, CD, CDS, RI, CNIP, ARIL, MdD) =====
    if "union pour la démocratie française" in labels or "union pour la democratie francaise" in labels:
        return "liberal_center_right"
    if re.search(r"\budf\b", labels):
        return "liberal_center_right"
    if "mouvement réformateur" in labels or "mouvement reformateur" in labels:
        return "liberal_center_right"
    if "centre démocratie et progrès" in labels or "centre democratie et progres" in labels:
        return "liberal_center_right"
    if "centre des démocrates sociaux" in labels or "centre des democrates sociaux" in labels:
        return "liberal_center_right"
    if re.search(r"\bcds\b", labels):
        return "liberal_center_right"
    if "centre démocrate" in labels or "centre democrate" in labels:
        return "liberal_center_right"
    if "républicains indépendants" in labels or "republicains independants" in labels:
        return "liberal_center_right"
    if "alliance républicaine" in labels or "alliance republicaine" in labels:
        return "liberal_center_right"
    if "centre national des indépendants" in labels or "centre national des independants" in labels:
        return "liberal_center_right"
    if re.search(r"\bcnip\b", labels):
        return "liberal_center_right"
    if "mouvement des démocrates" in labels or "mouvement des democrates" in labels:
        return "liberal_center_right"
    if "réformateur" in labels or "reformateur" in labels:  # 1973-78 Mouvement réformateur and 1993 variants
        return "liberal_center_right"
    if "parti républicain" in labels or "parti republicain" in labels:  # PR, composante UDF
        return "liberal_center_right"
    if "démocratie chrétienne" in labels or "democratie chretienne" in labels:
        return "liberal_center_right"
    if "parti radical" in labels:  # Parti radical (valoisien, distinct des Radicaux de gauche déjà capturés)
        return "liberal_center_right"

    return "other"


# Doc-topic matrix

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
    pivot.columns = [f"topic_{int(c)}" for c in pivot.columns]
    return pivot


def doc_metadata(df: pd.DataFrame, doc_col: str, meta_cols: List[str]) -> pd.DataFrame:
    cols = [c for c in meta_cols if c in df.columns]
    if not cols:
        return pd.DataFrame({doc_col: df[doc_col].unique()})
    return df[[doc_col] + cols].groupby(doc_col, as_index=False).first()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build doc-level topic vectors and assign party families."
    )
    parser.add_argument("--include-bertopic-outliers", action="store_true",
                        help="Keep topic -1 from BERTopic (default: drop).")
    parser.add_argument("--lda-use-score", action="store_true",
                        help="Weight LDA chunk-topic counts by topic_lda_score.")

    args = parser.parse_args()

    print("\nLoading BERTopic chunks...")
    df_bert = pd.read_csv(BERTOPIC_CHUNKS_PATH)
    print(f"Chunks loaded: {len(df_bert)}")

    print("\nLoading LDA chunks...")
    df_lda = pd.read_csv(LDA_CHUNKS_PATH)
    print(f"Chunks loaded: {len(df_lda)}")

    if DOC_ID_COL not in df_bert.columns or BERTOPIC_TOPIC_COL not in df_bert.columns:
        raise ValueError("Missing columns in BERTopic chunks output.")
    if DOC_ID_COL not in df_lda.columns or LDA_TOPIC_COL not in df_lda.columns:
        raise ValueError("Missing columns in LDA chunks output.")


    drop_topics = [] if args.include_bertopic_outliers else [-1]
    bert_vec = build_doc_topic_matrix(
        df_bert,
        doc_col=DOC_ID_COL,
        topic_col=BERTOPIC_TOPIC_COL,
        weight_col=None,
        drop_topics=drop_topics,
    )

    bert_meta = doc_metadata(df_bert, DOC_ID_COL, DOC_META_COLS)
    bert_meta["party_family"] = bert_meta.apply(assign_party_family, axis=1)

    bert_out = bert_meta.merge(
        bert_vec.reset_index(),
        on=DOC_ID_COL,
        how="left",
    ).fillna(0.0)

    topic_cols_bert = [c for c in bert_out.columns if c.startswith("topic_")]
    if topic_cols_bert:
        bert_out["dominant_topic"] = (
            bert_out[topic_cols_bert].idxmax(axis=1).str.replace("topic_", "").astype(int)
        )

    bert_out.to_csv(BERTOPIC_VEC_PATH, index=False, encoding="utf-8-sig")


    weight_col = LDA_SCORE_COL if args.lda_use_score and LDA_SCORE_COL in df_lda.columns else None
    lda_vec = build_doc_topic_matrix(
        df_lda,
        doc_col=DOC_ID_COL,
        topic_col=LDA_TOPIC_COL,
        weight_col=weight_col,
        drop_topics=None,
    )

    lda_meta = doc_metadata(df_lda, DOC_ID_COL, DOC_META_COLS)
    lda_meta["party_family"] = lda_meta.apply(assign_party_family, axis=1)

    lda_out = lda_meta.merge(
        lda_vec.reset_index(),
        on=DOC_ID_COL,
        how="left",
    ).fillna(0.0)

    topic_cols_lda = [c for c in lda_out.columns if c.startswith("topic_")]
    if topic_cols_lda:
        lda_out["dominant_topic"] = (
            lda_out[topic_cols_lda].idxmax(axis=1).str.replace("topic_", "").astype(int)
        )

    lda_out.to_csv(LDA_VEC_PATH, index=False, encoding="utf-8-sig")


    family = (
        bert_meta[[DOC_ID_COL, "party_family"] + [c for c in DOC_META_COLS if c in bert_meta.columns]]
        .copy()
    )
    family.to_csv(PARTY_FAMILY_PATH, index=False, encoding="utf-8-sig")


    with open(INFO_PATH, "w", encoding="utf-8") as f:
        f.write("=== Doc-level Topic Vectors ===\n\n")
        f.write(f"BERTopic docs: {len(bert_out)}, topics: {len(topic_cols_bert)}\n")
        f.write(f"LDA docs: {len(lda_out)}, topics: {len(topic_cols_lda)}\n")
        f.write(f"BERTopic outliers dropped: {not args.include_bertopic_outliers}\n")
        f.write(f"LDA weighted by score: {args.lda_use_score}\n\n")

        f.write("=== Party family counts (BERTopic docs) ===\n")
        f.write(bert_out["party_family"].value_counts().to_string())
        f.write("\n\n")

        if "year" in bert_out.columns:
            f.write("=== Docs by year ===\n")
            f.write(bert_out["year"].value_counts().sort_index().to_string())
            f.write("\n\n")

        if "dominant_topic" in bert_out.columns:
            f.write("=== Dominant BERTopic topics ===\n")
            f.write(bert_out["dominant_topic"].value_counts().head(20).to_string())
            f.write("\n")


    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Party family counts
    counts = bert_out["party_family"].value_counts()
    colors = ["#4a7ab5"] * len(counts)
    counts.plot(kind="barh", ax=axes[0], color=colors)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Number of documents")
    axes[0].set_title("Documents per party family")
    for i, v in enumerate(counts.values):
        axes[0].text(v + max(counts.values) * 0.01, i, str(v), va="center", fontsize=9)

    # Crosstab year x family
    if "year" in bert_out.columns:
        ct = pd.crosstab(bert_out["year"], bert_out["party_family"])
        valid = [c for c in ct.columns if c not in ("unclassified", "other")]
        ct_valid = ct[valid] if valid else ct
        ct_valid.plot(kind="bar", stacked=True, ax=axes[1], colormap="tab10")
        axes[1].set_xlabel("Year")
        axes[1].set_ylabel("Number of documents")
        axes[1].set_title("Party family distribution by year")
        axes[1].legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)
        axes[1].tick_params(axis="x", rotation=0)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "party_family_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("\nDone.")
    print(f"Saved BERTopic doc vectors: {BERTOPIC_VEC_PATH}")
    print(f"Saved LDA doc vectors: {LDA_VEC_PATH}")
    print(f"Saved party_family table: {PARTY_FAMILY_PATH}")
    print(f"Saved figures: {FIG_DIR}")
    print(f"Saved summary: {INFO_PATH}")


if __name__ == "__main__":
    main()
