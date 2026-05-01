# Topic Modeling and Semantic Mapping of French Political Manifestos

**Authors**
- Brian Ramesh
- Clément Destouesse

Most commits are on Brian's account because the topic-modelling fits had to run on his machine (the only one with enough RAM/GPU for the embeddings and BERTopic). The two of us contributed equally to the design and code.


---

## Project

Topic modeling on the Archelec corpus of French legislative *professions de foi* from 1973 to 1993 (~21k documents). The question is whether the political families of the period can be cartographed in a semantic space inferred from text alone. The pipeline produces candidate-level thematic profiles and three downstream analyses: clustering by political family, thematic specialisation, and sentiment divergence on shared topics.

ENSAE NLP course (2025–2026), C. Kermorvant.

## Main references

Grootendorst, M. (2022). *BERTopic: Neural topic modeling with a class-based TF-IDF procedure.* arXiv:2203.05794.

## Repository

### Top-level layout

```text
Topic_modeling_nlp/
├── README.md
├── pyproject.toml
├── stop_word_fr.txt                # custom French stopword list (~700 entries)
├── data/                           # raw inputs (CSV + OCR text files per year)
├── outputs/                        # per-step outputs (CSV, figures, reports)
├── scripts/                        # numbered pipeline scripts (01 → 13)
└── ntbks/                          # exploratory notebooks
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
├── 09_doc_topic_vectors/           # doc_topic_vectors_bertopic.csv + doc_party_family.csv
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
├── 07_bertopic.py                  # BERTopic + reduce_topics(30) (HDBSCAN noise -1 kept)
├── 09_doc_topic_vectors.py         # Aggregate to doc level + party family mapping
├── 10_analyses.py                  # Inter-family clustering and thematic specialisation
├── 11_visualizations.py            # 2D projection, topic sanity digest, BERTopic native views
├── 12_sentiment.py                 # Sentiment on shared topics
└── 13_run_all.py                   # Orchestrator: runs steps 1 to 12 in order
```

The orchestrator script `13_run_all.py` runs steps 1 to 11 in order. To re-run a single step, just call it directly.
</details>


---

## Reproducing the results

Once the environment is set up (Step 1 below) and `data/archelect_search.csv` + the year folders are in place:

```bash
uv run python scripts/13_run_all.py     # full run, ~1h20 on M-series with MPS
```

Each step is a self-contained script. To re-run a single step, just call it directly: `uv run python scripts/07_bertopic.py`.

---

## Step-by-step

<details>
<summary><strong>Step 1 — Install dependencies</strong></summary>

The project is managed with [uv](https://docs.astral.sh/uv/). Dependencies are pinned in `uv.lock`.

```bash
cd Topic_modeling_nlp
uv sync                                 # creates .venv and installs from uv.lock
uv run python -m spacy download fr_core_news_sm
```

To run a script:
```bash
uv run python scripts/13_run_all.py
```

Main dependencies (declared in `pyproject.toml`):
- `bertopic`, `sentence-transformers`, `umap-learn`, `hdbscan`
- `gensim`, `spacy`, `transformers`, `torch`
- `scikit-learn`, `scipy`
- `pandas`, `numpy`, `matplotlib`, `plotly`, `kaleido`, `tqdm`
</details>

<details>
<summary><strong>Step 2 — Join metadata with OCR text files</strong></summary>

```bash
python scripts/01_data_load.py
```

Loads the metadata CSV, filters to the five legislative elections, indexes the year-by-year `.txt` files and joins them on the document id. Writes a few descriptive plots.

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

Computes a per-document OCR quality score (share of recognisable French words minus share of length-≤2 tokens) and drops the most degraded documents.

Outputs:
- `outputs/02_data_quality/corpus_cleaned.csv`
- `outputs/02_data_quality/reports/ocr_quality.txt`
- `outputs/02_data_quality/figures/ocr_score_by_year.png`, `ocr_score_distribution.png`, `kept_removed_by_year.png`
</details>

<details>
<summary><strong>Step 4 — Two-track preprocessing</strong></summary>

```bash
python scripts/03_preprocessing.py
```

Builds two versions of each document:

- `text_clean` — light cleaning only (OCR artefact removal, generic phrase removal, normalisation). Goes to the sentence transformer.
- `text_preprocessed` — lemmatised (spaCy `fr_core_news_sm`) then stopword-pruned with our custom 700-word list. Goes to the c-TF-IDF representation of BERTopic.

The order matters: **lemmatise first, then remove stopwords**, otherwise inflected variants slip through.

Outputs:
- `outputs/03_preprocessing/corpus_preprocessed.csv`
- `outputs/03_preprocessing/reports/preprocessing_info.txt`
- `outputs/03_preprocessing/figures/token_reduction.png`
</details>

<details>
<summary><strong>Step 5 — Family-level EDA</strong></summary>

```bash
python scripts/04_eda.py
```

Top words per political family and the share of a few political keywords (`travail`, `emploi`, `immigration`, `securite`, `europe`...) across families and years. Runs on `text_clean` (post-cleaning), so OCR boilerplate and generic campaign phrases are already gone.

Outputs:
- `outputs/04_eda/eda_top_words_by_family.csv`
- `outputs/04_eda/eda_keyword_trends_by_family.csv`
- `outputs/04_eda/figures/eda_top_words_by_family.png`, `eda_keyword_trends_by_family.png`
</details>

<details>
<summary><strong>Step 6 — Chunking strategy comparison (one-off)</strong></summary>

```bash
python scripts/05_chunking_eval.py
```

One-shot comparison of three chunking strategies (fixed length, sentence, paragraph) on a sample of 200 documents, scored by `c_v` coherence and topic diversity. Used to justify the fixed-length 150/overlap 30/min 40 picked up by step 7.

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

Splits `text_clean` into overlapping windows of 150 words (overlap 30, min size 40) and embeds each chunk with `paraphrase-multilingual-MiniLM-L12-v2`. The script auto-detects MPS / CUDA / CPU.

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

Configuration:

- UMAP: `n_neighbors=30`, `n_components=5`, cosine, `random_state=42`
- HDBSCAN: `min_cluster_size=60`, `eom`
- CountVectorizer: French stopwords list, `min_df=2`, `max_df=0.5`, `ngram_range=(1, 2)`
- `reduce_topics(nr_topics=30)`, `reduce_outliers` off. The HDBSCAN noise label `-1` is kept as a "no thematic cluster" bucket (~54% of chunks). Step 9 drops these chunks before family-level aggregation.

We tested the four combinations of `K ∈ {20, 30}` and `reduce_outliers ∈ {on, off}`. With `reduce_outliers=on`, 40–90% of chunks pile into a single mixed-family cluster. K=20 fuses PCF and LO into one cluster. K=30 with outliers off keeps a dedicated topic for each major peripheral family (FN, PCF, LO, écologistes) plus three smaller ones (POE, PLN, PFN).

Topics are labelled by hand in `ntbks/topic_labelling.ipynb` (top words + 5 chunks from 5 distinct docs, then a label and family hint go into `outputs/07_bertopic/topic_labels.csv`).

Outputs:
- `outputs/07_bertopic/chunks_with_topics.csv`
- `outputs/07_bertopic/topic_info.csv`
- `outputs/07_bertopic/topic_labels.csv` (manual labels, one per topic)
- `outputs/07_bertopic/models/bertopic_model/`
- `outputs/07_bertopic/reports/topic_modeling_info.txt`
- `outputs/07_bertopic/figures/topic_barchart.png`, `topic_barchart_labelled.png`, `topic_heatmap.png`, `topic_hierarchy.png`, `topic_sizes.png`
</details>

<details>
<summary><strong>Step 9 — Document-level aggregation and party family mapping</strong></summary>

```bash
python scripts/09_doc_topic_vectors.py
```

Aggregates the chunk-level topic assignments back to the document level (probability vector over topics per doc), and maps the 1,800+ raw `titulaire-soutien` / `titulaire-liste` values to seven political families (`national_right`, `radical_left`, `communist_left`, `socialist_left`, `ecologist`, `gaullist_right`, `liberal_center_right`) plus the residuals `unclassified` and `other`. The ordered rule set is in `scripts/_family_mapping.py`.

Outputs:
- `outputs/09_doc_topic_vectors/doc_topic_vectors_bertopic.csv`
- `outputs/09_doc_topic_vectors/doc_party_family.csv`
- `outputs/09_doc_topic_vectors/reports/doc_topic_vectors_info.txt`
- `outputs/09_doc_topic_vectors/figures/party_family_distribution.png`
</details>

<details>
<summary><strong>Step 10 — Inter-family clustering and thematic specialisation</strong></summary>

```bash
python scripts/10_analyses.py
```

- **Clustering**: KMeans (k=7) on documents in two spaces — the mean-embedding profile and the topic-distribution profile — scored by Purity, ARI, NMI, Silhouette. A 3-panel UMAP projection of the chunks is also produced.
- **Specialisation**: lift, Pearson χ², Cramér's V on the family × topic contingency table, plus the top-3 specialised topics per family.

Outputs (under `outputs/10_analyses/`):
- `clustering/party_clustering_embedding.csv`, `clustering/party_clustering_topic.csv`
- `figures/clustering/party_projection.png`
- `reports/clustering/party_clustering_info.txt`
- `specialisation/topic_specialization_lift.csv`, `topic_specialization_top.csv`
- `figures/specialisation/topic_specialization_heatmap.png`
- `reports/specialisation/topic_specialization_chi2.txt`, `topic_specialization_info.txt`
</details>

<details>
<summary><strong>Step 11 — Topic sanity digest</strong></summary>

```bash
python scripts/11_visualizations.py
```

Per-topic digest used for the manual labelling step: top c-TF-IDF terms, three representative chunks, and three heuristic flags (`GENERIC_BOILERPLATE`, `REPETITIVE_WORDS`, `DUPLICATE_CHUNKS`). The duplicate-chunks flag is what surfaces the centrally-distributed-templates pattern discussed in the report.

Outputs:
- `outputs/11_visualizations/sanity/topic_flags.csv`
- `outputs/11_visualizations/sanity/reports/topic_digest.txt`
</details>

<details>
<summary><strong>Step 12 — Sentiment on shared topics</strong></summary>

```bash
python scripts/12_sentiment.py
```

Each chunk gets scored by `cmarkea/distilcamembert-base-sentiment`, with the 5-star output mapped to a continuous score in [-1, 1] via the expected rating. For every "shared" topic — at least 3 families with ≥10 chunks each — we report the range and std of family-mean sentiment, plus extracts of the most positive / most negative chunks per polarised topic. Scores are cached in `chunks_with_sentiment.csv`; delete the file to rescore from scratch.

Outputs (under `outputs/12_sentiment/`):
- `chunks_with_sentiment.csv` (cache)
- `sentiment_by_party_topic.csv`
- `shared_topics_polarization.csv`
- `reports/qualitative_extracts.txt`, `reports/sentiment_info.txt`
- `figures/sentiment_heatmap.png`, `polarized_topics_box.png`
</details>

---

### Hardware and runtime

Approximate runtime on Apple M-series with MPS acceleration:

| Step | Time |
|---|---|
| 01 → 03 | ~5 min |
| 03 (lemmatisation) | ~8 min |
| 06 (embedding ~118k chunks) | ~8 min |
| 07 (BERTopic + reduce_topics) | ~25 min |
| 09 → 11 | ~5 min |
| 12 (sentiment, first run) | ~30 min, cached afterwards |
| Total (cold full run) | ~1h20 |
