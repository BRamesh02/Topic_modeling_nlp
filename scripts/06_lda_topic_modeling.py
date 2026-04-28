"""
Step 6 — LDA Topic Modeling

Inputs:
data/corpus_chunks.csv

Outputs:
data/chunks_with_topics_lda.csv
outputs/lda_topic_info.csv
outputs/lda_modeling_info.txt
models/lda_model/
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from gensim.corpora import Dictionary
from gensim.models import LdaModel, CoherenceModel
from gensim.utils import simple_preprocess


# =====================
# Paths
# =====================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_DIR = PROJECT_ROOT / "models" / "lda_model"

CHUNKS_PATH = DATA_DIR / "corpus_chunks.csv"

OUTPUT_CHUNKS_PATH = DATA_DIR / "chunks_with_topics_lda.csv"
TOPIC_INFO_PATH = OUTPUT_DIR / "lda_topic_info.csv"
INFO_PATH = OUTPUT_DIR / "lda_modeling_info.txt"

MODEL_PATH = MODEL_DIR / "lda_model.gensim"
DICT_PATH = MODEL_DIR / "dictionary.gensim"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# =====================
# Parameters
# =====================

TEXT_COL = "chunk_text"

NUM_TOPICS = 20
PASSES = 10
ITERATIONS = 100
RANDOM_STATE = 42

MIN_DF = 5
MAX_DF = 0.5
TOPN_WORDS = 15


# =====================
# Helpers
# =====================

def tokenize(text: str) -> List[str]:
    return simple_preprocess(text, deacc=True, min_len=2)


def build_corpus(texts: List[str]) -> tuple[List[List[str]], Dictionary, List[List[tuple[int, int]]]]:
    tokenized = [tokenize(t) for t in texts]

    dictionary = Dictionary(tokenized)
    dictionary.filter_extremes(no_below=MIN_DF, no_above=MAX_DF)

    corpus = [dictionary.doc2bow(t) for t in tokenized]
    return tokenized, dictionary, corpus


def dominant_topic(doc_topics: List[tuple[int, float]]) -> tuple[int, float]:
    if not doc_topics:
        return -1, 0.0
    topic_id, score = max(doc_topics, key=lambda x: x[1])
    return int(topic_id), float(score)


# =====================
# Main
# =====================

def main() -> None:
    print("\nLoading chunks...")
    df = pd.read_csv(CHUNKS_PATH)
    print(f"Chunks loaded: {len(df)}")

    if TEXT_COL not in df.columns:
        raise ValueError(f"Missing column: {TEXT_COL}")

    texts = df[TEXT_COL].fillna("").astype(str).tolist()

    print("\nBuilding corpus...")
    tokenized, dictionary, corpus = build_corpus(texts)

    if len(dictionary) == 0:
        raise ValueError("Empty dictionary after filtering. Adjust MIN_DF/MAX_DF.")

    print("\nFitting LDA...")
    lda_model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=NUM_TOPICS,
        passes=PASSES,
        iterations=ITERATIONS,
        random_state=RANDOM_STATE,
        eval_every=None,
        alpha="auto",
        eta="auto",
    )

    # =====================
    # Topic assignment
    # =====================

    print("\nAssigning topics to chunks...")
    topic_ids = []
    topic_scores = []

    for bow in corpus:
        doc_topics = lda_model.get_document_topics(bow, minimum_probability=0.0)
        topic_id, score = dominant_topic(doc_topics)
        topic_ids.append(topic_id)
        topic_scores.append(score)

    df["topic_lda"] = topic_ids
    df["topic_lda_score"] = topic_scores

    # =====================
    # Topic info
    # =====================

    topic_rows = []
    for topic_id in range(NUM_TOPICS):
        terms = lda_model.show_topic(topic_id, topn=TOPN_WORDS)
        words = [w for w, _ in terms]
        weights = [float(wt) for _, wt in terms]
        topic_rows.append(
            {
                "topic_id": topic_id,
                "top_words": " ".join(words),
                "top_weights": " ".join(f"{w:.4f}" for w in weights),
            }
        )

    topic_info = pd.DataFrame(topic_rows)

    # =====================
    # Coherence
    # =====================

    print("\nComputing coherence...")
    topics_words = [row["top_words"].split() for row in topic_rows]
    coherence_model = CoherenceModel(
        topics=topics_words,
        texts=tokenized,
        dictionary=dictionary,
        coherence="c_v",
    )
    coherence_score = float(coherence_model.get_coherence())

    # =====================
    # Save outputs
    # =====================

    print("\nSaving outputs...")
    df.to_csv(OUTPUT_CHUNKS_PATH, index=False, encoding="utf-8-sig")
    topic_info.to_csv(TOPIC_INFO_PATH, index=False, encoding="utf-8-sig")

    lda_model.save(str(MODEL_PATH))
    dictionary.save(str(DICT_PATH))

    with open(INFO_PATH, "w", encoding="utf-8") as f:
        f.write("=== LDA Topic Modeling Info ===\n\n")
        f.write(f"Number of chunks: {len(df)}\n")
        f.write(f"Number of topics: {NUM_TOPICS}\n")
        f.write(f"Coherence (c_v): {coherence_score:.4f}\n\n")

        f.write("=== Parameters ===\n")
        f.write(f"MIN_DF: {MIN_DF}\n")
        f.write(f"MAX_DF: {MAX_DF}\n")
        f.write(f"PASSES: {PASSES}\n")
        f.write(f"ITERATIONS: {ITERATIONS}\n")
        f.write(f"RANDOM_STATE: {RANDOM_STATE}\n")
        f.write(f"TOPN_WORDS: {TOPN_WORDS}\n\n")

        f.write("=== Top Topics ===\n")
        f.write(topic_info.head(30).to_string(index=False))

    print("\nDone.")
    print(f"Saved chunks with topics: {OUTPUT_CHUNKS_PATH}")
    print(f"Saved topic info: {TOPIC_INFO_PATH}")
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved dictionary: {DICT_PATH}")
    print(f"Saved summary: {INFO_PATH}")


if __name__ == "__main__":
    main()
