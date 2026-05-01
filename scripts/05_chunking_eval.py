"""
Step 5 — Compare three chunking strategies (sentence / paragraph / fixed-length)
on a sample of text_clean. Reports c_v coherence and topic diversity per
strategy. Used to justify the fixed-length choice picked up by step 6.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from gensim.models.coherencemodel import CoherenceModel
from gensim.utils import simple_preprocess


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = PROJECT_ROOT / "outputs"

PREV_DIR = OUTPUTS / "03_preprocessing"
STEP_DIR = OUTPUTS / "05_chunking_eval"
REPORTS_DIR = STEP_DIR / "reports"
FIG_DIR = STEP_DIR / "figures"

STEP_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

INPUT_PATH = PREV_DIR / "corpus_preprocessed.csv"


TEXT_COL = "text_clean"
SAMPLE = 200
MIN_WORDS = 20
FIXED_SIZES = [100, 150, 200, 300]
OVERLAP = 30
TOPN = 10
RANDOM_STATE = 42

OUTPUT_CSV = STEP_DIR / "chunking_methods_eval.csv"
OUTPUT_TXT = REPORTS_DIR / "chunking_methods_eval.txt"


def chunk_by_sentence(text, min_words):
    if not isinstance(text, str) or not text.strip():
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if len(p.split()) >= min_words]


def is_heading_line(line, max_words):
    words = line.split()
    if not words or len(words) > max_words:
        return False
    if line.isupper():
        return True
    if re.search(r"[.!?;:]$", line):
        return False
    return True


def join_lines(lines):
    if not lines:
        return ""
    out = lines[0]
    for line in lines[1:]:
        if out.endswith("-"):
            out = out[:-1] + line.lstrip()
        else:
            out = out + " " + line
    return re.sub(r"\s+", " ", out).strip()


def chunk_by_paragraph(text, min_words, heading_max_words=8):
    if not isinstance(text, str) or not text.strip():
        return []

    lines = [l.strip() for l in text.splitlines()]
    paragraphs = []
    buffer = []

    for line in lines:
        if not line:
            if buffer:
                paragraphs.append(join_lines(buffer))
                buffer = []
            continue
        if is_heading_line(line, heading_max_words):
            if buffer:
                paragraphs.append(join_lines(buffer))
                buffer = []
            paragraphs.append(line)
            continue
        buffer.append(line)

    if buffer:
        paragraphs.append(join_lines(buffer))

    return [p for p in paragraphs if len(p.split()) >= min_words]


def chunk_by_fixed_length(text, chunk_size, min_words, overlap=0):
    if not isinstance(text, str) or not text.strip():
        return []
    words = text.split()
    if len(words) < min_words:
        return []
    chunks = []
    step = max(1, chunk_size - overlap)
    for start in range(0, len(words), step):
        chunk_words = words[start: start + chunk_size]
        if len(chunk_words) >= min_words:
            chunks.append(" ".join(chunk_words))
        if start + chunk_size >= len(words):
            break
    return chunks


def tokenize(text):
    return simple_preprocess(text, deacc=True, min_len=2)


def build_topics_and_texts(chunks, topn):
    tokenized = [tokenize(c) for c in chunks]
    tokenized = [t for t in tokenized if t]

    dictionary = Dictionary(tokenized)
    if len(dictionary) == 0:
        return [], tokenized, dictionary

    corpus = [dictionary.doc2bow(t) for t in tokenized]
    tfidf = TfidfModel(corpus)

    topics = []
    for doc in corpus:
        tfidf_doc = tfidf[doc]
        if not tfidf_doc:
            continue
        top_terms = sorted(tfidf_doc, key=lambda x: x[1], reverse=True)[:topn]
        topic_words = [dictionary[word_id] for word_id, _ in top_terms]
        if len(topic_words) >= 2:
            topics.append(topic_words)

    return topics, tokenized, dictionary


def compute_stats(chunks, topn):
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


def main():
    print(f"Loading {INPUT_PATH}...")
    df = pd.read_csv(INPUT_PATH)
    df = df[df[TEXT_COL].notna()].copy()

    if len(df) > SAMPLE:
        df = df.sample(n=SAMPLE, random_state=RANDOM_STATE)

    texts = df[TEXT_COL].astype(str).tolist()
    print(f"Documents used: {len(texts)}")

    results = []

    sentence_chunks = []
    for text in texts:
        sentence_chunks.extend(chunk_by_sentence(text, MIN_WORDS))
    results.append({"method": "sentence", **compute_stats(sentence_chunks, TOPN)})

    paragraph_chunks = []
    for text in texts:
        paragraph_chunks.extend(chunk_by_paragraph(text, MIN_WORDS))
    results.append({"method": "paragraph", **compute_stats(paragraph_chunks, TOPN)})

    for size in FIXED_SIZES:
        chunks_no_overlap = []
        for text in texts:
            chunks_no_overlap.extend(chunk_by_fixed_length(text, size, MIN_WORDS, 0))
        results.append({"method": f"fixed_{size}", **compute_stats(chunks_no_overlap, TOPN)})

        chunks_overlap = []
        for text in texts:
            chunks_overlap.extend(chunk_by_fixed_length(text, size, MIN_WORDS, OVERLAP))
        results.append({"method": f"fixed_{size}_ov{OVERLAP}",
                        **compute_stats(chunks_overlap, TOPN)})

    df_res = pd.DataFrame(results)
    df_res["combined_score"] = df_res["coherence_c_v"] * df_res["topic_diversity"]
    df_sorted = df_res.sort_values(by="combined_score", ascending=False)

    print("\nChunking methods evaluation")
    print(df_sorted.to_string(index=False))

    df_sorted.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"Saved CSV: {OUTPUT_CSV}")

    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        f.write("=== CHUNKING METHODS EVALUATION ===\n\n")
        f.write(f"Documents used: {len(texts)}\n")
        f.write(f"Min words per chunk: {MIN_WORDS}\n")
        f.write(f"Fixed sizes: {FIXED_SIZES} (overlap variant: {OVERLAP})\n")
        f.write(f"Topn words per chunk: {TOPN}\n\n")
        f.write(df_sorted.to_string(index=False))
        f.write("\n")

    print(f"Saved TXT: {OUTPUT_TXT}")

    # Figures
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    valid = df_sorted.dropna(subset=["coherence_c_v", "topic_diversity"])

    for _, r in valid.iterrows():
        color = "#c44e52" if "fixed" in str(r["method"]) else "#4a7ab5"
        axes[0].scatter(r["coherence_c_v"], r["topic_diversity"], s=80, color=color)
        axes[0].annotate(str(r["method"]), (r["coherence_c_v"], r["topic_diversity"]),
                         fontsize=8, xytext=(5, 5), textcoords="offset points")
    axes[0].set_xlabel("Coherence (c_v)")
    axes[0].set_ylabel("Topic diversity")
    axes[0].set_title("Chunking method: coherence vs topic diversity")
    axes[0].grid(alpha=0.3)

    df_plot = valid.sort_values("combined_score", ascending=True)
    axes[1].barh(range(len(df_plot)), df_plot["combined_score"].values, color="#4a7ab5")
    axes[1].set_yticks(range(len(df_plot)))
    axes[1].set_yticklabels(df_plot["method"].values, fontsize=9)
    axes[1].set_xlabel("Combined score (coherence × diversity)")
    axes[1].set_title("Chunking methods ranked")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "chunking_methods_comparison.png", dpi=150)
    plt.close()

    print(f"Saved figure: {FIG_DIR / 'chunking_methods_comparison.png'}")
    print("Done.")


if __name__ == "__main__":
    main()
