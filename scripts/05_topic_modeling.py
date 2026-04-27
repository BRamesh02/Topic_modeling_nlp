"""
Step 5 — BERTopic Topic Modeling

Inputs:
data/corpus_chunks.csv
data/chunk_embeddings.npy

Outputs:
data/chunks_with_topics.csv
outputs/topic_info.csv
outputs/topic_modeling_info.txt
models/bertopic_model/
figures/topic_barchart.html
figures/topic_heatmap.html
figures/topic_hierarchy.html

Optional PNG export requires:
pip install kaleido
"""

from pathlib import Path
import numpy as np
import pandas as pd
import torch

from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from hdbscan import HDBSCAN


# =====================
# Device
# =====================

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

print(f"Using device: {DEVICE}")
print("Note: BERTopic clustering runs mostly on CPU. The embeddings were already computed before.")


# =====================
# Paths
# =====================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIG_DIR = PROJECT_ROOT / "figures"
MODEL_DIR = PROJECT_ROOT / "models" / "bertopic_model"

CHUNKS_PATH = DATA_DIR / "corpus_chunks.csv"
EMBEDDINGS_PATH = DATA_DIR / "chunk_embeddings.npy"

OUTPUT_CHUNKS_PATH = DATA_DIR / "chunks_with_topics.csv"
TOPIC_INFO_PATH = OUTPUT_DIR / "topic_info.csv"
INFO_PATH = OUTPUT_DIR / "topic_modeling_info.txt"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)


# =====================
# Parameters
# =====================

TEXT_COL = "chunk_text"

MIN_TOPIC_SIZE = 150
N_NEIGHBORS = 15
N_COMPONENTS = 5
MIN_DIST = 0.0
RANDOM_STATE = 42


# =====================
# Helpers
# =====================

def save_plotly_figure(fig, html_path: Path, png_path: Path | None = None):
    fig.write_html(html_path)

    if png_path is not None:
        try:
            fig.write_image(png_path)
        except Exception as e:
            print(f"PNG export failed for {png_path.name}. Install kaleido if needed.")
            print(f"Error: {e}")


# =====================
# Main
# =====================

def main():
    print("\nLoading chunks...")
    df = pd.read_csv(CHUNKS_PATH)
    print(f"Chunks loaded: {len(df)}")

    if TEXT_COL not in df.columns:
        raise ValueError(f"Missing column: {TEXT_COL}")

    print("\nLoading embeddings...")
    embeddings = np.load(EMBEDDINGS_PATH)
    print(f"Embeddings shape: {embeddings.shape}")

    if len(df) != embeddings.shape[0]:
        raise ValueError(
            f"Mismatch: {len(df)} chunks but {embeddings.shape[0]} embeddings."
        )

    docs = df[TEXT_COL].fillna("").astype(str).tolist()

    # =====================
    # BERTopic components
    # =====================

    vectorizer_model = CountVectorizer(
        stop_words=None,
        min_df= 2, #5,
        ngram_range=(1, 2)
    )

    umap_model = UMAP(
        n_neighbors=N_NEIGHBORS,
        n_components=N_COMPONENTS,
        min_dist=MIN_DIST,
        metric="cosine",
        random_state=RANDOM_STATE
    )

    hdbscan_model = HDBSCAN(
        min_cluster_size=MIN_TOPIC_SIZE,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True
    )

    topic_model = BERTopic(
        language="multilingual",
        vectorizer_model=vectorizer_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        calculate_probabilities=False,
        verbose=True
    )

    # =====================
    # Fit model
    # =====================

    print("\nFitting BERTopic...")
    topics, _ = topic_model.fit_transform(docs, embeddings)

    df["topic"] = topics

    # =====================
    # Save outputs
    # =====================

    print("\nSaving topic assignments...")
    df.to_csv(OUTPUT_CHUNKS_PATH, index=False, encoding="utf-8-sig")

    topic_info = topic_model.get_topic_info()
    topic_info.to_csv(TOPIC_INFO_PATH, index=False, encoding="utf-8-sig")

    print("\nSaving BERTopic model...")
    topic_model.save(MODEL_DIR)

    # =====================
    # Visualizations
    # =====================

    print("\nSaving visualizations...")

    try:
        fig = topic_model.visualize_barchart(top_n_topics=20)
        save_plotly_figure(
            fig,
            FIG_DIR / "topic_barchart.html",
            FIG_DIR / "topic_barchart.png"
        )
    except Exception as e:
        print(f"Could not save topic barchart: {e}")

    try:
        fig = topic_model.visualize_heatmap()
        save_plotly_figure(
            fig,
            FIG_DIR / "topic_heatmap.html",
            FIG_DIR / "topic_heatmap.png"
        )
    except Exception as e:
        print(f"Could not save topic heatmap: {e}")

    try:
        fig = topic_model.visualize_hierarchy()
        save_plotly_figure(
            fig,
            FIG_DIR / "topic_hierarchy.html",
            FIG_DIR / "topic_hierarchy.png"
        )
    except Exception as e:
        print(f"Could not save topic hierarchy: {e}")

    # =====================
    # Summary
    # =====================

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
        f.write(f"RANDOM_STATE: {RANDOM_STATE}\n\n")

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