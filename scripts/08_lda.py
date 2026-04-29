"""
Step 8 — LDA baseline on text_preprocessed (lemmatised + stopwords removed),
chunked at the same granularity as step 6. Document identifiers match step 6
so that step 9 can aggregate the two models per doc. Run with --eval-k to do a
coherence-vs-K sweep instead of fitting the main model.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from gensim.corpora import Dictionary
from gensim.models import LdaModel, CoherenceModel
from gensim.utils import simple_preprocess
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = PROJECT_ROOT / "outputs"

PREV_DIR = OUTPUTS / "03_preprocessing"
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

NUM_TOPICS = 10
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


def get_topics_top_words(model: LdaModel, num_topics: int, n_top_words: int) -> List[List[str]]:
    return [
        [w for w, _ in model.show_topic(t, topn=n_top_words)]
        for t in range(num_topics)
    ]


def evaluate_k_coherence(
    tokenized: List[List[str]],
    dictionary: Dictionary,
    corpus: List,
    k_values: List[int],
    n_top_words: int = 10,
    passes: int = 5,
    iterations: int = 50,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    metrics = ["u_mass", "c_v", "c_uci", "c_npmi"]
    rows = []
    for k in k_values:
        print(f"\n[eval-k] Fitting LDA with k={k} ...")
        model = LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=k,
            passes=passes,
            iterations=iterations,
            random_state=random_state,
            eval_every=None,
            alpha="auto",
            eta="auto",
        )
        topics = get_topics_top_words(model, k, n_top_words)
        for met in metrics:
            cm = CoherenceModel(
                topics=topics,
                texts=tokenized,
                dictionary=dictionary,
                coherence=met,
            )
            score = float(cm.get_coherence())
            rows.append({"k": k, "metric": met, "coherence": score})
            print(f"  k={k}, {met}: {score:.4f}")
    return pd.DataFrame(rows)


def plot_coherence_curves(df: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    for met, sub in df.groupby("metric"):
        sub = sub.sort_values("k")
        scores = sub["coherence"].values
        rng = scores.max() - scores.min()
        norm = (scores - scores.min()) / rng if rng > 0 else np.zeros_like(scores)
        plt.plot(sub["k"].values, norm, marker="o", label=met)
    plt.xlabel("Number of topics (k)")
    plt.ylabel("Normalised coherence (per metric)")
    plt.title("LDA — coherence vs k (min–max normalised per metric)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="LDA topic modeling on the preprocessed corpus.")
    parser.add_argument(
        "--eval-k", action="store_true",
        help="Run coherence-vs-k analysis (no full fit). Saves coherence_eval.csv and a plot.",
    )
    parser.add_argument(
        "--k-values", type=int, nargs="+", default=[5, 10, 15, 20, 30],
        help="K values to test in --eval-k mode.",
    )
    parser.add_argument(
        "--eval-sample", type=int, default=80000,
        help="Number of chunks to sample for the coherence eval (0 = all). Eval is heavy: each k = 1 LDA fit.",
    )
    parser.add_argument(
        "--eval-passes", type=int, default=5,
        help="LDA passes during eval (lower than main fit for tractable runtime).",
    )
    args = parser.parse_args()

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

    if args.eval_k:
        print("\n=== Coherence-vs-k evaluation mode ===")
        if args.eval_sample and 0 < args.eval_sample < len(tokenized):
            rng = np.random.default_rng(RANDOM_STATE)
            idx = rng.choice(len(tokenized), size=args.eval_sample, replace=False)
            tokenized_eval = [tokenized[i] for i in idx]
            corpus_eval = [corpus[i] for i in idx]
            print(f"Sampled {len(tokenized_eval)} chunks for the evaluation.")
        else:
            tokenized_eval = tokenized
            corpus_eval = corpus

        eval_df = evaluate_k_coherence(
            tokenized_eval, dictionary, corpus_eval,
            k_values=args.k_values,
            n_top_words=10,
            passes=args.eval_passes,
            iterations=50,
        )
        eval_csv = STEP_DIR / "coherence_eval.csv"
        eval_df.to_csv(eval_csv, index=False, encoding="utf-8-sig")
        print(f"\nSaved coherence eval CSV: {eval_csv}")

        eval_fig = FIG_DIR / "coherence_vs_k.png"
        plot_coherence_curves(eval_df, eval_fig)
        print(f"Saved coherence eval figure: {eval_fig}")

        with open(REPORTS_DIR / "coherence_eval.txt", "w", encoding="utf-8") as f:
            f.write("=== LDA — coherence vs k ===\n\n")
            f.write(f"K values: {args.k_values}\n")
            f.write(f"Sample size: {len(tokenized_eval)} chunks\n")
            f.write(f"Eval passes: {args.eval_passes}\n\n")
            f.write("Best k per metric (highest raw coherence):\n")
            for met, sub in eval_df.groupby("metric"):
                best = sub.loc[sub["coherence"].idxmax()]
                f.write(f"  {met}: k={int(best['k'])}  (score={best['coherence']:.4f})\n")
            f.write("\nFull table:\n")
            f.write(eval_df.pivot(index="k", columns="metric", values="coherence").to_string())
        print("\nDone (eval-k mode). No full LDA fit was performed.")
        return

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
