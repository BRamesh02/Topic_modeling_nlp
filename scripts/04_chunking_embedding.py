"""
Step 4 — Chunking + Embeddings

Input:
data/corpus_preprocessed.csv

Outputs:
data/corpus_chunks.csv
data/chunk_embeddings.npy
outputs/chunking_embedding_info.txt
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import torch


# Paths

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

INPUT_PATH = DATA_DIR / "corpus_preprocessed.csv"
CHUNKS_PATH = DATA_DIR / "corpus_chunks.csv"
EMBEDDINGS_PATH = DATA_DIR / "chunk_embeddings.npy"
INFO_PATH = OUTPUT_DIR / "chunking_embedding_info.txt"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Parameters

TEXT_COL = "text_preprocessed"

CHUNK_SIZE = 150      # nombre de mots par chunk
CHUNK_OVERLAP = 30    # chevauchement entre chunks
MIN_CHUNK_WORDS = 40  # supprime les morceaux trop courts

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

device = "mps" if torch.backends.mps.is_available() else "cpu" #device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

model = SentenceTransformer(MODEL_NAME, device=device)


# Chunking

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


# Main

def main():
    print("Loading preprocessed corpus...")
    df = pd.read_csv(INPUT_PATH)
    print(f"Documents loaded: {len(df)}")

    if TEXT_COL not in df.columns:
        raise ValueError(f"Missing text column: {TEXT_COL}")

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
    print(f"Average chunks per document: {len(df_chunks) / len(df):.2f}")

    df_chunks.to_csv(CHUNKS_PATH, index=False, encoding="utf-8-sig")
    print(f"Saved chunks: {CHUNKS_PATH}")

    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    print("Computing embeddings...")
    embeddings = model.encode(
        df_chunks["chunk_text"].tolist(),
        batch_size=128 if device == "mps" else 64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    np.save(EMBEDDINGS_PATH, embeddings)
    print(f"Saved embeddings: {EMBEDDINGS_PATH}")
    print(f"Embedding shape: {embeddings.shape}")

    with open(INFO_PATH, "w", encoding="utf-8") as f:
        f.write("=== CHUNKING + EMBEDDING INFO ===\n\n")
        f.write(f"Input documents: {len(df)}\n")
        f.write(f"Chunks created: {len(df_chunks)}\n")
        f.write(f"Average chunks per document: {len(df_chunks) / len(df):.2f}\n\n")
        f.write(f"Chunk size: {CHUNK_SIZE}\n")
        f.write(f"Chunk overlap: {CHUNK_OVERLAP}\n")
        f.write(f"Minimum chunk words: {MIN_CHUNK_WORDS}\n\n")
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
    print("Done.")


if __name__ == "__main__":
    main()