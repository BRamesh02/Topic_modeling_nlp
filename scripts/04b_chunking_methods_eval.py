"""
Step 4b — Chunking Methods Evaluation

Input:
legislatives93/*.txt (default)

Outputs:
outputs/chunking_methods_eval.csv
outputs/chunking_methods_eval.txt
"""

from __future__ import annotations

from pathlib import Path
import argparse
import re
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from gensim.models.coherencemodel import CoherenceModel
from gensim.utils import simple_preprocess


# Paths

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
DEFAULT_INPUT_DIR = PROJECT_ROOT.parent / "legislatives81" / "text_files"/ "1981" / "legislatives"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Parameters

DEFAULT_MAX_FILES = 5
DEFAULT_MIN_WORDS = 20
DEFAULT_FIXED_SIZES = [100, 200, 300]  # test 2 or 3 sizes from CLI if needed
DEFAULT_TOPN = 10


# Chunking

def chunk_by_sentence(text: str, min_words: int) -> List[str]:
    if not isinstance(text, str) or not text.strip():
        return []

    parts = [p.strip() for p in text.split(".")]
    return [p for p in parts if len(p.split()) >= min_words]


def _is_heading_line(line: str, max_words: int) -> bool:
    words = line.split()
    if not words or len(words) > max_words:
        return False
    if line.isupper():
        return True
    if re.search(r"[.!?;:]$", line):
        return False
    return True


def _join_lines(lines: List[str]) -> str:
    if not lines:
        return ""

    out = lines[0]
    for line in lines[1:]:
        if out.endswith("-"):
            out = out[:-1] + line.lstrip()
        else:
            out = out + " " + line
    return re.sub(r"\s+", " ", out).strip()


def chunk_by_paragraph(text: str, min_words: int, heading_max_words: int = 8) -> List[str]:
    if not isinstance(text, str) or not text.strip():
        return []

    lines = [l.strip() for l in text.splitlines()]
    paragraphs: List[str] = []
    buffer: List[str] = []

    for line in lines:
        if not line:
            if buffer:
                paragraphs.append(_join_lines(buffer))
                buffer = []
            continue

        if _is_heading_line(line, max_words=heading_max_words):
            if buffer:
                paragraphs.append(_join_lines(buffer))
                buffer = []
            paragraphs.append(line)
            continue

        buffer.append(line)

    if buffer:
        paragraphs.append(_join_lines(buffer))

    return [p for p in paragraphs if len(p.split()) >= min_words]


def chunk_by_fixed_length(text: str, chunk_size: int, min_words: int) -> List[str]:
    if not isinstance(text, str) or not text.strip():
        return []

    words = text.split()
    if len(words) < min_words:
        return []

    chunks: List[str] = []
    for start in range(0, len(words), chunk_size):
        chunk_words = words[start : start + chunk_size]
        if len(chunk_words) >= min_words:
            chunks.append(" ".join(chunk_words))
    return chunks


# Evaluation

def tokenize(text: str) -> List[str]:
    # simple_preprocess removes punctuation and lowercases; deacc keeps ASCII output
    return simple_preprocess(text, deacc=True, min_len=2)


def build_topics_and_texts(chunks: Iterable[str], topn: int) -> Tuple[List[List[str]], List[List[str]], Dictionary]:
    tokenized = [tokenize(c) for c in chunks]
    tokenized = [t for t in tokenized if t]

    dictionary = Dictionary(tokenized)
    if len(dictionary) == 0:
        return [], tokenized, dictionary

    corpus = [dictionary.doc2bow(t) for t in tokenized]
    tfidf = TfidfModel(corpus)

    topics: List[List[str]] = []
    for doc in corpus:
        tfidf_doc = tfidf[doc]
        if not tfidf_doc:
            continue
        top_terms = sorted(tfidf_doc, key=lambda x: x[1], reverse=True)[:topn]
        topic_words = [dictionary[word_id] for word_id, _ in top_terms]
        if len(topic_words) >= 2:
            topics.append(topic_words)

    return topics, tokenized, dictionary


def compute_stats(chunks: List[str], topn: int) -> dict:
    if not chunks:
        return {
            "n_chunks": 0,
            "avg_chunk_words": np.nan,
            "coherence_c_v": np.nan,
            "topic_diversity": np.nan,
            "lexical_diversity": np.nan,
        }

    chunk_lengths = [len(c.split()) for c in chunks]
    avg_chunk_words = float(np.mean(chunk_lengths)) if chunk_lengths else np.nan

    topics, tokenized, dictionary = build_topics_and_texts(chunks, topn=topn)

    coherence_score = np.nan
    if topics and len(dictionary) > 0:
        cm = CoherenceModel(
            topics=topics,
            texts=tokenized,
            dictionary=dictionary,
            coherence="c_v",
        )
        coherence_score = float(cm.get_coherence())

    topic_diversity = np.nan
    if topics:
        unique_words = {w for topic in topics for w in topic}
        total_words = len(topics) * topn
        if total_words > 0:
            topic_diversity = len(unique_words) / total_words

    lexical_diversity = np.nan
    if tokenized:
        ratios = [len(set(t)) / len(t) for t in tokenized if t]
        lexical_diversity = float(np.mean(ratios)) if ratios else np.nan

    return {
        "n_chunks": len(chunks),
        "avg_chunk_words": avg_chunk_words,
        "coherence_c_v": coherence_score,
        "topic_diversity": topic_diversity,
        "lexical_diversity": lexical_diversity,
    }


# IO

def load_texts(input_dir: Path, max_files: int) -> List[str]:
    files = sorted(input_dir.glob("*.txt"))
    if max_files is not None and max_files > 0:
        files = files[:max_files]

    texts: List[str] = []
    for path in files:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if text.strip():
            texts.append(text)

    return texts


# Main

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate chunking methods on sample texts.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--max-files", type=int, default=DEFAULT_MAX_FILES)
    parser.add_argument("--min-words", type=int, default=DEFAULT_MIN_WORDS)
    parser.add_argument("--fixed-sizes", type=int, nargs="+", default=DEFAULT_FIXED_SIZES)
    parser.add_argument("--topn", type=int, default=DEFAULT_TOPN)
    parser.add_argument("--output-csv", type=Path, default=OUTPUT_DIR / "chunking_methods_eval.csv")
    parser.add_argument("--output-txt", type=Path, default=OUTPUT_DIR / "chunking_methods_eval.txt")

    args = parser.parse_args()

    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input dir not found: {args.input_dir}")

    texts = load_texts(args.input_dir, args.max_files)
    if not texts:
        raise ValueError("No input texts loaded. Check input directory and file encoding.")

    results = []

    # Sentence
    sentence_chunks: List[str] = []
    for text in texts:
        sentence_chunks.extend(chunk_by_sentence(text, min_words=args.min_words))
    stats = compute_stats(sentence_chunks, topn=args.topn)
    results.append({"method": "sentence_dot", **stats})

    # Paragraph
    paragraph_chunks: List[str] = []
    for text in texts:
        paragraph_chunks.extend(chunk_by_paragraph(text, min_words=args.min_words))
    stats = compute_stats(paragraph_chunks, topn=args.topn)
    results.append({"method": "paragraph_newline", **stats})

    # Fixed length
    for size in args.fixed_sizes:
        fixed_chunks: List[str] = []
        for text in texts:
            fixed_chunks.extend(
                chunk_by_fixed_length(text, chunk_size=size, min_words=args.min_words)
            )
        stats = compute_stats(fixed_chunks, topn=args.topn)
        results.append({"method": f"fixed_{size}", **stats})

    df = pd.DataFrame(results)

    # Simple combined score (higher is better) to compare quickly
    df["combined_score"] = df["coherence_c_v"] * df["topic_diversity"]

    df_sorted = df.sort_values(by="combined_score", ascending=False)

    print("Chunking methods evaluation")
    print(df_sorted.to_string(index=False))

    df_sorted.to_csv(args.output_csv, index=False, encoding="utf-8-sig")
    print(f"Saved CSV: {args.output_csv}")

    with open(args.output_txt, "w", encoding="utf-8") as f:
        f.write("=== CHUNKING METHODS EVALUATION ===\n\n")
        f.write(f"Input dir: {args.input_dir}\n")
        f.write(f"Files loaded: {len(texts)}\n")
        f.write(f"Min words per chunk: {args.min_words}\n")
        f.write(f"Fixed sizes: {args.fixed_sizes}\n")
        f.write(f"Topn words per chunk: {args.topn}\n\n")
        f.write(df_sorted.to_string(index=False))
        f.write("\n")

    print(f"Saved TXT: {args.output_txt}")
    print("Done.")


if __name__ == "__main__":
    main()
