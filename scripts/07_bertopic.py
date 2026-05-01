"""
Step 7 — Fit BERTopic on the precomputed chunk embeddings. UMAP + HDBSCAN for
the clustering, CountVectorizer (with the curated French stopword list) for the
c-TF-IDF representation, then reduce_topics(30). HDBSCAN noise (-1) is kept
as-is rather than reassigned, so step 9 will drop it before family-level
aggregation.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import torch

from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from hdbscan import HDBSCAN


if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

print(f"Using device: {DEVICE}")


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = PROJECT_ROOT / "outputs"

PREV_DIR = OUTPUTS / "06_chunking_embedding"
STEP_DIR = OUTPUTS / "07_bertopic"
FIG_DIR = STEP_DIR / "figures"
REPORTS_DIR = STEP_DIR / "reports"
MODEL_DIR = STEP_DIR / "models" / "bertopic_model"

STEP_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)

STOPWORDS_PATH = PROJECT_ROOT / "stop_word_fr.txt"

CHUNKS_PATH = PREV_DIR / "corpus_chunks.csv"
EMBEDDINGS_PATH = PREV_DIR / "chunk_embeddings.npy"

OUTPUT_CHUNKS_PATH = STEP_DIR / "chunks_with_topics.csv"
TOPIC_INFO_PATH = STEP_DIR / "topic_info.csv"
INFO_PATH = REPORTS_DIR / "topic_modeling_info.txt"


TEXT_COL = "chunk_text"

MIN_TOPIC_SIZE = 60
N_NEIGHBORS = 30
N_COMPONENTS = 5
MIN_DIST = 0.0
RANDOM_STATE = 42

VECT_MIN_DF = 2
VECT_MAX_DF = 0.5
NGRAM_RANGE = (1, 2)
NR_TOPICS = 30


def load_stopwords(path):
    with open(path, "r", encoding="utf-8") as f:
        words = [line.strip().lower() for line in f if line.strip()]
    return sorted(set(words))


def main():
    print("\nLoading chunks...")
    df = pd.read_csv(CHUNKS_PATH)
    print(f"Chunks loaded: {len(df)}")

    print("\nLoading embeddings...")
    embeddings = np.load(EMBEDDINGS_PATH)
    print(f"Embeddings shape: {embeddings.shape}")

    docs = df[TEXT_COL].fillna("").astype(str).tolist()

    french_stopwords = load_stopwords(STOPWORDS_PATH)

    vectorizer_model = CountVectorizer(
        stop_words=french_stopwords,
        min_df=VECT_MIN_DF,
        max_df=VECT_MAX_DF,
        ngram_range=NGRAM_RANGE,
    )

    umap_model = UMAP(
        n_neighbors=N_NEIGHBORS,
        n_components=N_COMPONENTS,
        min_dist=MIN_DIST,
        metric="cosine",
        random_state=RANDOM_STATE,
    )

    hdbscan_model = HDBSCAN(
        min_cluster_size=MIN_TOPIC_SIZE,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )

    topic_model = BERTopic(
        language="multilingual",
        vectorizer_model=vectorizer_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        calculate_probabilities=False,
        verbose=True,
    )

    print("\nFitting BERTopic...")
    topics, _ = topic_model.fit_transform(docs, embeddings)
    print(f"Outliers: {sum(t == -1 for t in topics)} / {len(topics)}")

    topic_model.reduce_topics(docs, nr_topics=NR_TOPICS)
    topics = list(topic_model.topics_)
    df["topic"] = topics

    print("\nSaving topic assignments...")
    df.to_csv(OUTPUT_CHUNKS_PATH, index=False, encoding="utf-8-sig")

    topic_info = topic_model.get_topic_info()
    topic_info.to_csv(TOPIC_INFO_PATH, index=False, encoding="utf-8-sig")

    print("\nSaving BERTopic model...")
    topic_model.save(str(MODEL_DIR), serialization="pickle", save_ctfidf=True)

    print("\nSaving visualizations...")

    fig = topic_model.visualize_barchart(top_n_topics=20)
    fig.write_image(FIG_DIR / "topic_barchart.png", width=1600, height=1000, scale=2)

    fig = topic_model.visualize_heatmap()
    fig.write_image(FIG_DIR / "topic_heatmap.png", width=1200, height=1000, scale=2)

    fig = topic_model.visualize_hierarchy()
    fig.write_image(FIG_DIR / "topic_hierarchy.png", width=1400, height=1000, scale=2)

    import matplotlib.pyplot as plt
    sizes = topic_info[topic_info["Topic"] != -1].sort_values("Count", ascending=False).head(30)
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(sizes)), sizes["Count"].values, color="#4a7ab5")
    plt.xticks(range(len(sizes)), sizes["Topic"].astype(int).values, rotation=0, fontsize=8)
    plt.xlabel("Topic id")
    plt.ylabel("Number of chunks")
    plt.title("Top 30 topics by size (chunks)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "topic_sizes.png", dpi=150)
    plt.close()

    n_topics = len(set(topics)) - (1 if -1 in topics else 0)
    n_outliers = sum(t == -1 for t in topics)
    outlier_rate = n_outliers / len(df)

    with open(INFO_PATH, "w", encoding="utf-8") as f:
        f.write("=== BERTopic Modeling Info ===\n\n")
        f.write(f"Device detected: {DEVICE}\n")
        f.write(f"Number of chunks: {len(df)}\n")
        f.write(f"Number of topics excluding outliers: {n_topics}\n")
        f.write(f"Outliers: {n_outliers}\n")
        f.write(f"Outlier rate: {outlier_rate:.3f}\n\n")

        f.write("=== Parameters ===\n")
        f.write(f"MIN_TOPIC_SIZE: {MIN_TOPIC_SIZE}\n")
        f.write(f"N_NEIGHBORS: {N_NEIGHBORS}\n")
        f.write(f"N_COMPONENTS: {N_COMPONENTS}\n")
        f.write(f"MIN_DIST: {MIN_DIST}\n")
        f.write(f"RANDOM_STATE: {RANDOM_STATE}\n")
        f.write(f"VECT_MIN_DF: {VECT_MIN_DF}\n")
        f.write(f"VECT_MAX_DF: {VECT_MAX_DF}\n")
        f.write(f"NGRAM_RANGE: {NGRAM_RANGE}\n")
        f.write(f"Stopwords loaded: {len(french_stopwords) if french_stopwords else 0}\n\n")

        f.write("=== Top Topics ===\n")
        f.write(topic_info.head(30).to_string(index=False))

    print("\nDone.")
    print(f"Saved chunks with topics: {OUTPUT_CHUNKS_PATH}")
    print(f"Saved topic info: {TOPIC_INFO_PATH}")
    print(f"Saved model: {MODEL_DIR}")
    print(f"Saved summary: {INFO_PATH}")
    print(f"Saved figures: {FIG_DIR}")


if __name__ == "__main__":
    main()
