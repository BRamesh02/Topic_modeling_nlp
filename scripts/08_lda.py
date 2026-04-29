"""
Step 8 — LDA Topic Modeling

Runs LDA on text_preprocessed (lemmatized + stopwords removed) chunked at the
same granularity as step 6, but on the preprocessed column directly. Document
identifiers (doc_id) match step 6 so step 9 can aggregate both models per doc.

Inputs:
data/corpus_preprocessed.csv  (column text_preprocessed)

Outputs:
data/chunks_with_topics_lda.csv
outputs/lda_topic_info.csv
outputs/lda_modeling_info.txt
models/lda_model/
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd
from gensim.corpora import Dictionary
from gensim.models import LdaModel, CoherenceModel
from gensim.utils import simple_preprocess
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = PROJECT_ROOT / "outputs"

PREV_DIR = OUTPUTS / "04_preprocessing"
STEP_DIR = OUTPUTS / "08_lda"
REPORTS_DIR = STEP_DIR / "reports"
FIG_DIR = STEP_DIR / "figures"
MODEL_DIR = STEP_DIR / "models" / "lda_model"

STEP_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

INPUT_PATH = PREV_DIR / "corpus_preprocessed.csv"

OUTPUT_CHUNKS_PATH = STEP_DIR / "chunks_with_topics_lda.csv"
TOPIC_INFO_PATH = STEP_DIR / "lda_topic_info.csv"
INFO_PATH = REPORTS_DIR / "lda_modeling_info.txt"

MODEL_PATH = MODEL_DIR / "lda_model.gensim"
DICT_PATH = MODEL_DIR / "dictionary.gensim"


TEXT_COL = "text_preprocessed"
DOC_ID_COL = "id"

CHUNK_SIZE = 150
CHUNK_OVERLAP = 30
MIN_CHUNK_WORDS = 40

NUM_TOPICS = 20
PASSES = 10
ITERATIONS = 100
RANDOM_STATE = 42

MIN_DF = 5
MAX_DF = 0.5
TOPN_WORDS = 15

METADATA_COLS = [
    "id", "date", "year", "contexte-tour",
    "titulaire-liste", "titulaire-soutien",
    "titulaire-sexe", "titulaire-prenom", "titulaire-nom",
]


def split_into_chunks(text: str, chunk_size: int, overlap: int, min_words: int) -> List[str]:
    if not isinstance(text, str) or not text.strip():
        return []
    words = text.split()
    if len(words) < min_words:
        return []
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        if len(chunk_words) >= min_words:
            chunks.append(" ".join(chunk_words))
        if end >= len(words):
            break
        start += chunk_size - overlap
    return chunks


def tokenize(text: str) -> List[str]:
    return simple_preprocess(text, deacc=True, min_len=2)


def dominant_topic(doc_topics: List[tuple[int, float]]) -> tuple[int, float]:
    if not doc_topics:
        return -1, 0.0
    topic_id, score = max(doc_topics, key=lambda x: x[1])
    return int(topic_id), float(score)


def main() -> None:
    print("\nLoading preprocessed corpus...")
    df = pd.read_csv(INPUT_PATH)
    print(f"Documents loaded: {len(df)}")

    if TEXT_COL not in df.columns:
        raise ValueError(f"Missing column: {TEXT_COL}")

    metadata_cols = [c for c in METADATA_COLS if c in df.columns]

    print("\nChunking text_preprocessed...")
    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Chunking"):
        chunks = split_into_chunks(
            row[TEXT_COL],
            chunk_size=CHUNK_SIZE,
            overlap=CHUNK_OVERLAP,
            min_words=MIN_CHUNK_WORDS,
        )
        for i, chunk in enumerate(chunks):
            new_row = {col: row[col] for col in metadata_cols}
            new_row["doc_id"] = row[DOC_ID_COL]
            new_row["chunk_id"] = f"{row[DOC_ID_COL]}_lda_chunk_{i:03d}"
            new_row["chunk_index"] = i
            new_row["chunk_text"] = chunk
            new_row["chunk_words"] = len(chunk.split())
            rows.append(new_row)

    df_chunks = pd.DataFrame(rows)
    print(f"Chunks created: {len(df_chunks)}")

    print("\nBuilding dictionary and corpus...")
    tokenized = [tokenize(t) for t in df_chunks["chunk_text"].tolist()]
    dictionary = Dictionary(tokenized)
    dictionary.filter_extremes(no_below=MIN_DF, no_above=MAX_DF)

    if len(dictionary) == 0:
        raise ValueError("Empty dictionary after filtering. Adjust MIN_DF/MAX_DF.")

    corpus = [dictionary.doc2bow(t) for t in tokenized]

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

    print("\nAssigning topics to chunks...")
    topic_ids = []
    topic_scores = []
    for bow in corpus:
        doc_topics = lda_model.get_document_topics(bow, minimum_probability=0.0)
        topic_id, score = dominant_topic(doc_topics)
        topic_ids.append(topic_id)
        topic_scores.append(score)

    df_chunks["topic_lda"] = topic_ids
    df_chunks["topic_lda_score"] = topic_scores

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

    print("\nComputing coherence...")
    topics_words = [row["top_words"].split() for row in topic_rows]
    coherence_model = CoherenceModel(
        topics=topics_words,
        texts=tokenized,
        dictionary=dictionary,
        coherence="c_v",
    )
    coherence_score = float(coherence_model.get_coherence())

    print("\nSaving outputs...")
    df_chunks.to_csv(OUTPUT_CHUNKS_PATH, index=False, encoding="utf-8-sig")
    topic_info.to_csv(TOPIC_INFO_PATH, index=False, encoding="utf-8-sig")

    lda_model.save(str(MODEL_PATH))
    dictionary.save(str(DICT_PATH))

    with open(INFO_PATH, "w", encoding="utf-8") as f:
        f.write("=== LDA Topic Modeling Info ===\n\n")
        f.write(f"Number of chunks: {len(df_chunks)}\n")
        f.write(f"Number of topics: {NUM_TOPICS}\n")
        f.write(f"Coherence (c_v): {coherence_score:.4f}\n\n")

        f.write("=== Parameters ===\n")
        f.write(f"CHUNK_SIZE: {CHUNK_SIZE} (overlap {CHUNK_OVERLAP}, min {MIN_CHUNK_WORDS})\n")
        f.write(f"MIN_DF: {MIN_DF}\n")
        f.write(f"MAX_DF: {MAX_DF}\n")
        f.write(f"PASSES: {PASSES}\n")
        f.write(f"ITERATIONS: {ITERATIONS}\n")
        f.write(f"RANDOM_STATE: {RANDOM_STATE}\n")
        f.write(f"TOPN_WORDS: {TOPN_WORDS}\n\n")

        f.write("=== Top Topics ===\n")
        f.write(topic_info.head(30).to_string(index=False))


    import matplotlib.pyplot as plt
    import numpy as np

    # Top words per topic (grid)
    cols = 4
    rows = (NUM_TOPICS + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 2.4 * rows), squeeze=False)
    for tid in range(NUM_TOPICS):
        terms = lda_model.show_topic(tid, topn=10)
        words = [w for w, _ in terms][::-1]
        weights = [float(wt) for _, wt in terms][::-1]
        ax = axes[tid // cols][tid % cols]
        ax.barh(range(len(words)), weights, color="#4a7ab5")
        ax.set_yticks(range(len(words)))
        ax.set_yticklabels(words, fontsize=8)
        ax.set_title(f"Topic {tid}", fontsize=10)
        ax.tick_params(axis="x", labelsize=7)
    for tid in range(NUM_TOPICS, rows * cols):
        axes[tid // cols][tid % cols].axis("off")
    plt.suptitle(f"LDA — Top 10 words per topic (coherence c_v = {coherence_score:.3f})", fontsize=12, y=1.0)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "lda_top_words_per_topic.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Topic prevalence (number of chunks per dominant topic)
    topic_counts = pd.Series(topic_ids).value_counts().sort_index()
    plt.figure(figsize=(10, 5))
    plt.bar(topic_counts.index, topic_counts.values, color="#4a7ab5")
    plt.xlabel("Topic id")
    plt.ylabel("Number of chunks (dominant topic)")
    plt.title("LDA — chunks per dominant topic")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "lda_topic_prevalence.png", dpi=150)
    plt.close()

    print("\nDone.")
    print(f"Saved chunks with topics: {OUTPUT_CHUNKS_PATH}")
    print(f"Saved topic info: {TOPIC_INFO_PATH}")
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved figures: {FIG_DIR}")
    print(f"Saved summary: {INFO_PATH}")


if __name__ == "__main__":
    main()
