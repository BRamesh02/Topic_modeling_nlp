# Topic Modeling and Semantic Mapping of French Political Manifestos

**Authors**
- Brian Ramesh
- Clément Destouesse

---

## Project Overview

Electoral manifestos provide a structured and comparable form of political communication, making them well-suited for computational analysis. This repository contains an end-to-end NLP pipeline that applies modern topic modeling techniques to the **Archelec corpus** of French legislative *professions de foi* between 1973 and 1993, with the goal of mapping political parties and candidates in a shared semantic space inferred from text alone.

The project covers the full pipeline from raw OCR text and metadata to candidate-level thematic profiles, hypothesis tests on inter-party clustering and specialisation, and a sentiment analysis on shared topics.

The current repository implements:

- A two-track preprocessing strategy (light cleaning for embeddings, lemmatised + stopword-pruned for c-TF-IDF and LDA)
- Sentence-transformer embeddings on overlapping 150-word chunks
- BERTopic with tuned hyperparameters (UMAP + HDBSCAN + class-based TF-IDF, outlier reassignment, manual topic-count reduction)
- LDA as a methodological cross-check
- Document-level aggregation, family mapping (7 political families), 2D projections
- A four-step cartographic analysis: topic interpretability (LDA + BERTopic convergence), inter-family clustering (Purity / ARI / NMI on document profiles), thematic specialisation (chi-square + Cramér's V on the family × topic table), and tonal divergence on shared topics (sentiment range)

This project was conducted as part of the ENSAE course **Natural Language Processing (2025–2026)**, given by C. Kermorvant. The full project report can be found in the `report/` folder.

---

## References

This project follows the BERTopic methodology introduced by Grootendorst (2022), with additional comparative analyses against an LDA baseline and complementary sentiment scoring.

> Grootendorst, M. (2022). *BERTopic: Neural topic modeling with a class-based TF-IDF procedure.* arXiv:2203.05794
>
> Blei, D., Ng, A., & Jordan, M. (2003). *Latent Dirichlet Allocation.* Journal of Machine Learning Research, 3, 993–1022.
>
> Stoltz, D., & Taylor, M. (2021). *Cultural cartography with word embeddings.* Poetics.
>
> Gonthier, F. (2024). *Qui veut réformer la démocratie ? Une analyse computationnelle des professions de foi électorales en France (2015–2024).*

---

## Code Provenance and Originality

This project is primarily original code written for research and experimentation. The data joining, OCR quality assessment, two-track preprocessing pipeline, chunking strategy, hypothesis test suite, validation digest, sentiment scoring projection, and orchestration scripts were implemented specifically for this repository. Standard open-source libraries are used as-is for the heavy lifting:

- `bertopic` for the embedding-based topic model
- `sentence-transformers` (model `paraphrase-multilingual-MiniLM-L12-v2`) for chunk embeddings
- `umap-learn` and `hdbscan` for dimensionality reduction and clustering
- `gensim` for the LDA baseline and coherence scoring
- `transformers` (model `cmarkea/distilcamembert-base-sentiment`) for chunk-level sentiment scoring
- `spacy` (model `fr_core_news_sm`) for lemmatisation
- `scikit-learn` for KMeans, PCA, t-SNE, and the clustering metrics
- `scipy` for the chi-square independence test
- `pandas`, `numpy`, `matplotlib`, `plotly`, `kaleido` for data handling and figures

---

## Repository Structure

### Global overview

```text
Topic_modeling_nlp/
├── README.md                       # Project overview, structure, and usage
├── pyproject.toml                  # Project metadata and dependencies
├── stop_word_fr.txt                # Custom French stopword list (~700 entries)
├── data/                           # Raw inputs only (CSV + OCR text files per year)
├── outputs/                        # Per-step outputs (CSV, figures, reports)
├── scripts/                        # 13 numbered pipeline scripts (01 → 13)
├── report/                         # NeurIPS-style LaTeX report
└── ntbks/                          # Exploratory notebooks
```

<details>
<summary><strong>data/</strong></summary>

```text
├── archelect_search.csv            # Archelec metadata (all elections, ~33k rows)
├── 1973/                           # OCR text files for 1973 elections
├── 1978/                           # OCR text files for 1978 elections
├── 1981/                           # OCR text files for 1981 elections
├── 1988/                           # OCR text files for 1988 elections
└── 1993/                           # OCR text files for 1993 elections
```

The `data/` folder contains only raw inputs. All intermediate artefacts go to `outputs/NN_<step>/`.
</details>

<details>
<summary><strong>outputs/</strong></summary>

Each pipeline step writes to its own folder following the convention `outputs/NN_<step>/`, with a homogeneous structure:

- CSV files at the root of the step folder
- `figures/` subfolder for all PNG figures
- `reports/` subfolder for all TXT reports

```text
├── 01_data_load/                   # corpus_joined.csv + reports/ + figures/
├── 02_data_quality/                # corpus_cleaned.csv + OCR diagnostics
├── 03_preprocessing/               # corpus_preprocessed.csv (text_clean + text_preprocessed)
├── 04_eda/                         # top words by year/party/family + keyword trends (run on text_clean)
├── 05_chunking_eval/               # comparison of chunking methods
├── 06_chunking_embedding/          # corpus_chunks.csv + chunk_embeddings.npy
├── 07_bertopic/                    # chunks_with_topics.csv + topic_info.csv + models/
├── 08_lda/                         # chunks_with_topics_lda.csv + lda_topic_info.csv + models/
├── 09_doc_topic_vectors/           # doc_topic_vectors_bertopic/lda.csv + doc_party_family.csv
├── 10_analyses/                    # Clustering and specialisation analyses with figures and reports
├── 11_visualizations/              # PCA/t-SNE projection, sanity digest, native BERTopic views
└── 12_sentiment/                   # Sentiment scoring on shared topics + qualitative extracts
```
</details>

<details>
<summary><strong>scripts/</strong></summary>

```text
├── 01_data_load.py                 # Join CSV metadata with OCR text files, descriptive stats
├── 02_data_quality.py              # OCR quality scoring + filter
├── 03_preprocessing.py             # Light cleaning + lemmatisation + stopwords
├── 04_eda.py                       # Top words by year/party/family + keyword trends (on text_clean)
├── 05_chunking_eval.py             # Compare chunking strategies (fixed/sentence/paragraph)
├── 06_chunking_embedding.py        # Chunk text_clean + compute MiniLM embeddings
├── 07_bertopic.py                  # BERTopic + reduce_outliers + reduce_topics(20)
├── 08_lda.py                       # LDA baseline (10 topics)
├── 09_doc_topic_vectors.py         # Aggregate to doc level + party family mapping
├── 10_analyses.py                  # Inter-family clustering and thematic specialisation
├── 11_visualizations.py            # 2D projection, topic sanity digest, BERTopic native views
├── 12_sentiment.py                 # Sentiment on shared topics
└── 13_run_all.py                   # Orchestrator: run all 12 steps end-to-end
```

The orchestrator script `13_run_all.py` exposes flags `--from`, `--to`, `--skip`, `--continue-on-error`, and `--dry-run` for partial re-runs.
</details>

<details>
<summary><strong>report/</strong></summary>

```text
├── main.tex                        # NeurIPS template entry point
├── neurips_2026.sty                # Style file
├── references.bib                  # Bibliography
└── documents/
    ├── p2_related_work.tex         # State of the art
    ├── p3_data.tex                 # Corpus and preprocessing
    ├── p4_method.tex               # BERTopic theory + LDA baseline
    ├── p5_implementation.tex       # Hyperparameter tuning + topic coherence validation
    ├── p6_results.tex              # Cartographic analysis: interpretability, clustering, specialisation, tonality
    ├── p7_discussion.tex           # Limitations and incidental findings
    └── p8_conclusion.tex           # Conclusion
```
</details>

<details>
<summary><strong>ntbks/</strong></summary>

```text
└── nlp-lab-topic.ipynb             # Exploratory notebook (early prototyping)
```
</details>

---

## Global Summary to Reproduce All Results

To reproduce all results in one go (assuming the environment is set up and the raw data is in `data/`):

```bash
# Run the full pipeline (12 steps, ~1h30 on M-series with MPS)
python scripts/13_run_all.py
```

Or partially:

```bash
# Skip the heavy BERTopic re-fit (uses cached topic model)
python scripts/13_run_all.py --skip 7

# Resume from a given step
python scripts/13_run_all.py --from 9

# Dry-run (print plan without executing)
python scripts/13_run_all.py --dry-run
```

The orchestrator stops at the first failing step by default; use `--continue-on-error` to push through.

---

## Usage step by step to reproduce results

<details>
<summary><strong>Step 1 — Install dependencies</strong></summary>

```bash
cd Topic_modeling_nlp
python3 -m venv .venvnlp
source .venvnlp/bin/activate
pip install --upgrade pip
pip install -e .
python -m spacy download fr_core_news_sm
```

Main dependencies (declared in `pyproject.toml`):
- `bertopic`, `sentence-transformers`, `umap-learn`, `hdbscan`
- `gensim`, `spacy`, `transformers`
- `scikit-learn`, `scipy`
- `pandas`, `numpy`, `matplotlib`, `plotly`, `kaleido`, `tqdm`
</details>

<details>
<summary><strong>Step 2 — Join metadata with OCR text files</strong></summary>

```bash
python scripts/01_data_load.py
```

This step:
- Loads `data/archelect_search.csv`
- Filters to legislative elections of 1973, 1978, 1981, 1988, 1993
- Indexes all `.txt` files under `data/<year>/` and joins by document identifier
- Computes descriptive statistics

Outputs:
- `outputs/01_data_load/corpus_joined.csv`
- `outputs/01_data_load/reports/data_info.txt`
- `outputs/01_data_load/figures/documents_per_year.png`, `top_parties.png`, `text_length_distribution.png`
</details>

<details>
<summary><strong>Step 3 — OCR quality assessment</strong></summary>

```bash
python scripts/02_data_quality.py
```

Per-document OCR quality score (combining the share of recognisable French words and the share of short tokens), filtering of degraded documents, lexical diagnostics.

Outputs:
- `outputs/02_data_quality/corpus_cleaned.csv`
- `outputs/02_data_quality/reports/ocr_quality.txt`, `lexical_diagnostics.txt`
- `outputs/02_data_quality/figures/ocr_score_by_year.png`, `text_length_by_year.png`, etc.
</details>

<details>
<summary><strong>Step 4 — Two-track preprocessing</strong></summary>

```bash
python scripts/03_preprocessing.py
```

Produces two parallel versions of each document:

- `text_clean`: light cleaning (recurrent OCR artefacts, generic electoral phrases, normalisation). Used as input to the sentence transformer.
- `text_preprocessed`: lemmatised via spaCy + custom stopword removal. Used by LDA and the c-TF-IDF representation of BERTopic.

The order **lemmatise → stopwords** is essential to match inflected forms.

Outputs:
- `outputs/03_preprocessing/corpus_preprocessed.csv`
- `outputs/03_preprocessing/reports/preprocessing_info.txt`
- `outputs/03_preprocessing/figures/token_reduction.png`
</details>

<details>
<summary><strong>Step 5 — Exploratory data analysis</strong></summary>

```bash
python scripts/04_eda.py
```

Top words by year, by raw party, and by political family, plus keyword trend tracking on a shortlist of political terms. Runs on the `text_clean` column of `corpus_preprocessed.csv` (after step 3) so the OCR boilerplate and generic campaign phrases are already stripped.

Outputs:
- `outputs/04_eda/eda_top_words_by_year.csv`, `eda_top_words_by_party.csv`, `eda_top_words_by_family.csv`
- `outputs/04_eda/eda_keyword_trends.csv`, `eda_keyword_trends_by_family.csv`
- `outputs/04_eda/figures/eda_keyword_trends.png`, `eda_keyword_trends_by_family.png`, `eda_top_words_by_family.png`
</details>

<details>
<summary><strong>Step 6 — Chunking strategy comparison (one-off)</strong></summary>

```bash
python scripts/05_chunking_eval.py
```

Compares fixed-length, sentence, and paragraph chunking on a sample of 200 documents. Reports topic coherence (c_v) and topic diversity for each strategy. Justifies the choice of fixed-length 150 / overlap 30 / min 40 used in step 7.

Outputs:
- `outputs/05_chunking_eval/chunking_methods_eval.csv`
- `outputs/05_chunking_eval/reports/chunking_methods_eval.txt`
- `outputs/05_chunking_eval/figures/chunking_methods_comparison.png`
</details>

<details>
<summary><strong>Step 7 — Chunking and embeddings</strong></summary>

```bash
python scripts/06_chunking_embedding.py
```

Splits each document into overlapping chunks of 150 words (overlap 30, minimum 40) on `text_clean`, then encodes each chunk with `paraphrase-multilingual-MiniLM-L12-v2` on MPS / CUDA / CPU.

Outputs:
- `outputs/06_chunking_embedding/corpus_chunks.csv`
- `outputs/06_chunking_embedding/chunk_embeddings.npy`
- `outputs/06_chunking_embedding/reports/chunking_embedding_info.txt`
- `outputs/06_chunking_embedding/figures/chunks_distribution.png`
</details>

<details>
<summary><strong>Step 8 — BERTopic</strong></summary>

```bash
python scripts/07_bertopic.py
```

Pipeline:
- UMAP (`n_neighbors=30`, `n_components=5`, cosine, `random_state=42`)
- HDBSCAN (`min_cluster_size=60`, `eom`)
- CountVectorizer with French stopwords, `min_df=2`, `max_df=0.5`, `ngram_range=(1, 2)`
- `reduce_outliers(strategy="embeddings")` + `update_topics`
- `reduce_topics(nr_topics=30)`

Outputs:
- `outputs/07_bertopic/chunks_with_topics.csv`
- `outputs/07_bertopic/topic_info.csv`
- `outputs/07_bertopic/models/bertopic_model/`
- `outputs/07_bertopic/reports/topic_modeling_info.txt`
- `outputs/07_bertopic/figures/topic_barchart.png`, `topic_heatmap.png`, `topic_hierarchy.png`, `topic_documents.png`, `topic_sizes.png`
</details>

<details>
<summary><strong>Step 9 — LDA baseline</strong></summary>

```bash
python scripts/08_lda.py
```

20 topics, 10 passes, 100 iterations on the lemmatised version of the corpus. Coherence score c_v computed on the top 15 words per topic.

Outputs:
- `outputs/08_lda/chunks_with_topics_lda.csv`
- `outputs/08_lda/lda_topic_info.csv`
- `outputs/08_lda/models/lda_model/`
- `outputs/08_lda/reports/lda_modeling_info.txt`
- `outputs/08_lda/figures/lda_top_words_per_topic.png`, `lda_topic_prevalence.png`
</details>

<details>
<summary><strong>Step 10 — Document-level aggregation and party family mapping</strong></summary>

```bash
python scripts/09_doc_topic_vectors.py
```

Aggregates chunk-level topic assignments into document-level topic profiles. Maps the 1,800+ raw party labels to seven political families (`national_right`, `radical_left`, `communist_left`, `socialist_left`, `ecologist`, `gaullist_right`, `liberal_center_right`) via an ordered rule-based system, plus `unclassified` and `other` for residual cases.

Outputs:
- `outputs/09_doc_topic_vectors/doc_topic_vectors_bertopic.csv`
- `outputs/09_doc_topic_vectors/doc_topic_vectors_lda.csv`
- `outputs/09_doc_topic_vectors/doc_party_family.csv`
- `outputs/09_doc_topic_vectors/reports/doc_topic_vectors_info.txt`
- `outputs/09_doc_topic_vectors/figures/party_family_distribution.png`
</details>

<details>
<summary><strong>Step 11 — Inter-family clustering and thematic specialisation</strong></summary>

```bash
python scripts/10_analyses.py --all
# or selectively:
python scripts/10_analyses.py --analysis clustering
python scripts/10_analyses.py --analysis specialisation
```

Computes:

- **Clustering**: KMeans with k=7 in two representations (mean embedding, topic distribution) and metrics Purity, ARI, NMI, Silhouette.
- **Specialisation**: Lift, chi-square test, Cramér's V on the family × topic contingency table; top-3 specialised topics per family.

Outputs (under `outputs/10_analyses/`):
- `clustering/party_clustering_*.csv`, `figures/clustering/party_projection.png`, `reports/clustering/party_clustering_info.txt`
- `specialisation/topic_specialization_*.csv`, `figures/specialisation/topic_specialization_heatmap.png`, `reports/specialisation/topic_specialization_*.txt`
</details>

<details>
<summary><strong>Step 12 — Visualisations</strong></summary>

```bash
python scripts/11_visualizations.py --all
# or selectively:
python scripts/11_visualizations.py --viz projection
python scripts/11_visualizations.py --viz sanity
python scripts/11_visualizations.py --viz native
```

Three independent visualisation routines:

- `projection`: 2D PCA / t-SNE of document topic profiles, faceted by year, coloured by political family.
- `sanity`: digest of BERTopic topics with top c-TF-IDF terms, three representative chunks per topic, and automated heuristic flags (`GENERIC_BOILERPLATE`, `REPETITIVE_WORDS`, `DUPLICATE_CHUNKS`) used to validate topic coherence in our unsupervised setting.
- `native`: BERTopic native descriptive views `topics_per_class` and `topics_over_time`.

Outputs (under `outputs/11_visualizations/`):
- `projection/doc_topic_projection.csv`, `projection/figures/doc_projection*.png`
- `sanity/topic_flags.csv`, `sanity/reports/topic_digest.txt`
- `native/topics_per_class.csv`, `native/topics_over_time.csv`, `native/figures/*.png`
</details>

<details>
<summary><strong>Step 13 — Sentiment on shared topics (Brian's individual contribution)</strong></summary>

```bash
python scripts/12_sentiment.py
# Force re-scoring (slow):
python scripts/12_sentiment.py --force-rescore
```

Each chunk is scored with `cmarkea/distilcamembert-base-sentiment` and projected to a continuous score in [-1, 1] via the expected rating. Topics shared by at least three families with at least ten chunks each are kept; for each shared topic, the range and standard deviation of the family-mean sentiment are reported. Qualitative extracts (most positive vs most negative chunks) are saved for the most polarised topics. Sentiment scores are cached in `chunks_with_sentiment.csv` so subsequent runs do not re-score.

Outputs (under `outputs/12_sentiment/`):
- `chunks_with_sentiment.csv` (cache)
- `sentiment_by_party_topic.csv`
- `sentiment_by_party_topic_year.csv`
- `shared_topics_polarization.csv`
- `reports/qualitative_extracts.txt`, `reports/sentiment_info.txt`
- `figures/sentiment_heatmap.png`, `polarized_topics_box.png`, `sentiment_by_year.png`
</details>

---

## Current Results

The repository ships with output artefacts for all 12 steps on the configured corpus (21,156 documents over five legislative elections).

### Headline findings

| Question | Indicator | Value |
|---|---|---|
| Are the induced topics substantively meaningful? | c_v coherence (LDA, 10 topics) + qualitative convergence with BERTopic | 0.66 |
| Do thematic profiles cluster by political family? | Purity (embedding profile, k=7) | 0.54 |
| Are some families thematically more distinctive than others? | Cramér's V (chi-square on family × topic, 30 topics) | 0.55 |
| When families share a topic, do they frame it the same way? | Max range of family-mean sentiment on shared topics | 0.9 on [-1, 1] |

### Side finding

A by-product of the validation procedure (Step 12, sanity digest) revealed that several political organisations (Lutte Ouvrière, Front National, ecologist movements, Parti des Travailleurs) circulate centrally produced campaign templates that local candidates sign verbatim. The chunk-duplication ratio within those topics is around 4 candidates per unique chunk text, against approximately 1 for governmental parties (PS, RPR, UDF). This regularity, observable directly in the topic representations without any additional metadata, suggests that topic models can be repurposed as indicators of organisational discipline.

### Hardware and runtime

Estimated runtime on Apple M-series with MPS acceleration (the configuration we used during development):

| Step | Approximate time |
|---|---|
| 01 → 03 | ~5 min |
| 04 (lemmatisation) | ~8 min |
| 06 (embedding ~118k chunks) | ~8 min |
| 07 (BERTopic + reduce_outliers + reduce_topics) | ~25 min |
| 08 (LDA, 10 passes) | ~8 min |
| 09 → 11 | ~5 min |
| 12 (sentiment, first run) | ~30 min (cached afterwards) |
| **Total** (cold full run) | ~1h30 |
