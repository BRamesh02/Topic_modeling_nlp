"""
Step 6 — Chunking + Embeddings

Chunks the natural-text version (text_clean) so the sentence-transformer receives
French sentences (not lemmatized stopword-stripped output). For BERTopic c-TF-IDF,
step 7 uses a CountVectorizer with French stopwords directly.

Input:
data/corpus_preprocessed.csv  (must contain text_clean column)

Outputs:
data/corpus_chunks.csv          (chunk_text = chunk of text_clean)
data/chunk_embeddings.npy
outputs/chunking_embedding_info.txt
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = PROJECT_ROOT / "outputs"

PREV_DIR = OUTPUTS / "04_preprocessing"
STEP_DIR = OUTPUTS / "06_chunking_embedding"
REPORTS_DIR = STEP_DIR / "reports"
FIG_DIR = STEP_DIR / "figures"

STEP_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

INPUT_PATH = PREV_DIR / "corpus_preprocessed.csv"
CHUNKS_PATH = STEP_DIR / "corpus_chunks.csv"
EMBEDDINGS_PATH = STEP_DIR / "chunk_embeddings.npy"
INFO_PATH = REPORTS_DIR / "chunking_embedding_info.txt"


TEXT_COL = "text_clean"  # natural French, no lemma, no stopword removal

CHUNK_SIZE = 150
CHUNK_OVERLAP = 30
MIN_CHUNK_WORDS = 40

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

print("Device:", DEVICE)


def split_into_chunks(text, chunk_size=150, overlap=30, min_words=40):
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


def main():
    print("Loading preprocessed corpus...")
    df = pd.read_csv(INPUT_PATH)
    print(f"Documents loaded: {len(df)}")

    if TEXT_COL not in df.columns:
        raise ValueError(f"Missing text column: {TEXT_COL}. Run 04_preprocessing.py first.")

    print("Creating chunks...")

    rows = []

    metadata_cols = [
        "id",
        "date",
        "year",
        "contexte-tour",
        "titulaire-liste",
        "titulaire-soutien",
        "titulaire-sexe",
        "titulaire-prenom",
        "titulaire-nom",
    ]
    metadata_cols = [c for c in metadata_cols if c in df.columns]

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Chunking"):
        chunks = split_into_chunks(
            row[TEXT_COL],
            chunk_size=CHUNK_SIZE,
            overlap=CHUNK_OVERLAP,
            min_words=MIN_CHUNK_WORDS,
        )
        for i, chunk in enumerate(chunks):
            new_row = {col: row[col] for col in metadata_cols}
            new_row["doc_id"] = row["id"]
            new_row["chunk_id"] = f"{row['id']}_chunk_{i:03d}"
            new_row["chunk_index"] = i
            new_row["chunk_text"] = chunk
            new_row["chunk_words"] = len(chunk.split())
            rows.append(new_row)

    df_chunks = pd.DataFrame(rows)

    print(f"Chunks created: {len(df_chunks)}")
    print(f"Average chunks per document: {len(df_chunks) / max(1, len(df)):.2f}")

    df_chunks.to_csv(CHUNKS_PATH, index=False, encoding="utf-8-sig")
    print(f"Saved chunks: {CHUNKS_PATH}")

    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME, device=DEVICE)

    print("Computing embeddings...")
    embeddings = model.encode(
        df_chunks["chunk_text"].tolist(),
        batch_size=128 if DEVICE != "cpu" else 32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    np.save(EMBEDDINGS_PATH, embeddings)
    print(f"Saved embeddings: {EMBEDDINGS_PATH}")
    print(f"Embedding shape: {embeddings.shape}")

    with open(INFO_PATH, "w", encoding="utf-8") as f:
        f.write("=== CHUNKING + EMBEDDING INFO ===\n\n")
        f.write(f"Input column: {TEXT_COL} (natural French, kept for embeddings)\n")
        f.write(f"Input documents: {len(df)}\n")
        f.write(f"Chunks created: {len(df_chunks)}\n")
        f.write(f"Average chunks per document: {len(df_chunks) / max(1, len(df)):.2f}\n\n")
        f.write(f"Chunk size: {CHUNK_SIZE}\n")
        f.write(f"Chunk overlap: {CHUNK_OVERLAP}\n")
        f.write(f"Minimum chunk words: {MIN_CHUNK_WORDS}\n\n")
        f.write(f"Device: {DEVICE}\n")
        f.write(f"Embedding model: {MODEL_NAME}\n")
        f.write(f"Embedding matrix shape: {embeddings.shape}\n\n")

        f.write("Chunk length distribution:\n")
        f.write(df_chunks["chunk_words"].describe().to_string())
        f.write("\n\n")

        if "year" in df_chunks.columns:
            f.write("Chunks by year:\n")
            f.write(df_chunks["year"].value_counts().sort_index().to_string())
            f.write("\n\n")

        if "titulaire-soutien" in df_chunks.columns:
            f.write("Top political supports by chunks:\n")
            f.write(df_chunks["titulaire-soutien"].value_counts().head(20).to_string())
            f.write("\n")

    print(f"Saved info: {INFO_PATH}")


    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Chunks per document
    chunks_per_doc = df_chunks.groupby("doc_id").size()
    axes[0].hist(chunks_per_doc.values, bins=40, color="#4a7ab5", alpha=0.85)
    axes[0].axvline(chunks_per_doc.median(), color="red", linestyle="--",
                    label=f"median = {int(chunks_per_doc.median())}")
    axes[0].set_xlabel("Chunks per document")
    axes[0].set_ylabel("Number of documents")
    axes[0].set_title(f"Chunks per document (n={len(df_chunks)} chunks total)")
    axes[0].legend()

    # Chunks per year
    if "year" in df_chunks.columns:
        per_year = df_chunks["year"].value_counts().sort_index()
        axes[1].bar(per_year.index.astype(str), per_year.values, color="#4a7ab5")
        axes[1].set_xlabel("Year")
        axes[1].set_ylabel("Number of chunks")
        axes[1].set_title("Chunks per election year")
        for i, v in enumerate(per_year.values):
            axes[1].text(i, v + max(per_year.values) * 0.01, str(v), ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "chunks_distribution.png", dpi=150)
    plt.close()

    print(f"Saved figure: {FIG_DIR / 'chunks_distribution.png'}")
    print("Done.")


if __name__ == "__main__":
    main()
