# Topic Modeling and Semantic Mapping of French Political Manifestos

**Authors**
- Brian Ramesh
- Clément Destouesse

Most commits are on Brian's account because the topic-modelling fits had to run on his machine (the only one with enough RAM/GPU for the embeddings and BERTopic). The two of us contributed equally to the design and code.


---

## What this project does

We apply topic modeling to the **Archelec corpus** of French legislative *professions de foi* from 1973 to 1993 (~21k documents) and ask whether political families of that period can be cartographed in a semantic space inferred from text alone. The pipeline goes from raw OCR text + metadata to candidate-level thematic profiles, then runs three downstream analyses: clustering by political family, thematic specialisation, and sentiment divergence on shared topics. The full report is in `report/`.

ENSAE NLP course (2025–2026), C. Kermorvant.

## Main references

Grootendorst, M. (2022). *BERTopic: Neural topic modeling with a class-based TF-IDF procedure.* arXiv:2203.05794.
Blei, D., Ng, A., & Jordan, M. (2003). *Latent Dirichlet Allocation.* JMLR 3, 993–1022.

## Repository

### Top-level layout

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
├── 07_bertopic.py                  # BERTopic + reduce_topics(30) (HDBSCAN noise -1 kept)
├── 08_lda.py                       # LDA baseline (10 topics, c_v sweep on K)
├── 09_doc_topic_vectors.py         # Aggregate to doc level + party family mapping
├── 10_analyses.py                  # Inter-family clustering and thematic specialisation
├── 11_visualizations.py            # 2D projection, topic sanity digest, BERTopic native views
├── 12_sentiment.py                 # Sentiment on shared topics
└── 13_run_all.py                   # Orchestrator: runs steps 1 to 12 in order
```

The orchestrator script `13_run_all.py` exposes flags `--from`, `--to`, `--skip`, `--continue-on-error`, and `--dry-run` for partial re-runs.
</details>


---

## Reproducing the results

Once the environment is set up (Step 1 below) and `data/archelect_search.csv` + the year folders are in place:

```bash
python scripts/13_run_all.py            # full run, ~1h30 on M-series with MPS
python scripts/13_run_all.py --skip 7   # skip BERTopic refit (use cached model)
python scripts/13_run_all.py --from 9   # resume from step 9
python scripts/13_run_all.py --dry-run  # print plan only
```

Pipeline stops at the first failing step. Add `--continue-on-error` to push through.

---

## Step-by-step

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

Computes a per-document OCR quality score (share of recognisable French words minus share of length-≤2 tokens), drops the most degraded documents, and saves a few diagnostic plots on the raw vocabulary.

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

Builds two versions of each document:

- `text_clean` — light cleaning only (OCR artefact removal, generic phrase removal, normalisation). Goes to the sentence transformer.
- `text_preprocessed` — lemmatised (spaCy `fr_core_news_sm`) then stopword-pruned with our custom 700-word list. Goes to LDA and to the c-TF-IDF of BERTopic.

The order matters: **lemmatise first, then remove stopwords**, otherwise inflected variants slip through.

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

Top words by year, by raw party label, and by consolidated family. Also tracks a shortlist of political keywords (`travail`, `emploi`, `immigration`, `securite`, `europe`...) across years and families. Runs on `text_clean` (post-cleaning), not on the raw text, so the OCR boilerplate and generic campaign phrases are already gone.

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
- `reduce_topics(nr_topics=30)` — but `reduce_outliers` is **off**. The HDBSCAN noise label `-1` is kept as a "no thematic cluster" bucket (~54% of chunks). Step 9 drops these chunks before family-level aggregation.

Why this config: we tested the four combinations of `K ∈ {20, 30}` and `reduce_outliers ∈ {on, off}`. With `reduce_outliers=on`, 40–90% of chunks pile into a single mixed-family cluster; the cartography becomes unusable. Without it, K=30 keeps a dedicated topic for each major peripheral family (FN, PCF, LO, écologistes) plus three small ones (POE, PLN, PFN). K=20 fuses PCF and LO into one cluster, which we don't want. Details in `report/documents/p5_implementation.tex`.

After the fit, topics are labelled by hand in `ntbks/topic_labelling.ipynb` (top words + 5 chunks from 5 distinct docs, then a label and family hint go into `outputs/07_bertopic/topic_labels.csv`).

Outputs:
- `outputs/07_bertopic/chunks_with_topics.csv`
- `outputs/07_bertopic/topic_info.csv`
- `outputs/07_bertopic/topic_labels.csv` (manual labels, one per topic)
- `outputs/07_bertopic/models/bertopic_model/`
- `outputs/07_bertopic/reports/topic_modeling_info.txt`
- `outputs/07_bertopic/figures/topic_barchart.png`, `topic_heatmap.png`, `topic_hierarchy.png`, `topic_documents.png`, `topic_sizes.png`
</details>

<details>
<summary><strong>Step 9 — LDA baseline</strong></summary>

```bash
python scripts/08_lda.py            # main fit, K=10
python scripts/08_lda.py --eval-k   # coherence sweep on K ∈ {5, 10, 15, 20, 30}
```

LDA on the lemmatised corpus, 10 topics, 10 passes, 100 iterations, asymmetric priors. K=10 is picked from a coherence sweep over `K ∈ {5, 10, 15, 20, 30}` on 80k chunks, with four metrics (`u_mass`, `c_v`, `c_uci`, `c_npmi`); three out of four peak at K=10. Final `c_v` = 0.55 on top-15 words.

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

Aggregates the chunk-level topic assignments back to the document level (probability vector over topics per doc), and maps the 1,800+ raw `titulaire-soutien` / `titulaire-liste` values to seven political families (`national_right`, `radical_left`, `communist_left`, `socialist_left`, `ecologist`, `gaullist_right`, `liberal_center_right`) plus the residuals `unclassified` and `other`. The ordered rule set is in `scripts/_family_mapping.py`.

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

- **Clustering**: KMeans (k=7) on documents in two spaces — the mean-embedding profile and the topic-distribution profile — scored by Purity, ARI, NMI, Silhouette.
- **Specialisation**: lift, Pearson χ², Cramér's V on the family × topic contingency table, plus the top-3 specialised topics per family.

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

Three independent routines:

- `projection`: 2D PCA / t-SNE of doc topic profiles, faceted by year, coloured by family.
- `sanity`: per-topic digest with top c-TF-IDF terms, three representative chunks, and three heuristic flags (`GENERIC_BOILERPLATE`, `REPETITIVE_WORDS`, `DUPLICATE_CHUNKS`). The `DUPLICATE_CHUNKS` flag is what surfaced the centrally-distributed-templates side finding.
- `native`: BERTopic's own `topics_per_class` and `topics_over_time` views.

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

Each chunk gets scored by `cmarkea/distilcamembert-base-sentiment` (5-star → continuous score in [-1, 1] via expected rating). For every "shared" topic — at least 3 families with ≥10 chunks each — we report the range and std of family-mean sentiment, plus extracts of the most positive / most negative chunks per polarised topic. Scores are cached in `chunks_with_sentiment.csv`; subsequent runs skip the (slow) classifier pass unless `--force-rescore` is passed.

Outputs (under `outputs/12_sentiment/`):
- `chunks_with_sentiment.csv` (cache)
- `sentiment_by_party_topic.csv`
- `sentiment_by_party_topic_year.csv`
- `shared_topics_polarization.csv`
- `reports/qualitative_extracts.txt`, `reports/sentiment_info.txt`
- `figures/sentiment_heatmap.png`, `polarized_topics_box.png`, `sentiment_by_year.png`
</details>

---

## Headline numbers

Corpus: 21,156 documents, 118,383 chunks, 5 elections (1973–1993), 7 political families consolidated from 1,800+ raw labels.

| Question | Metric | Value |
|---|---|---|
| Are the induced topics substantively meaningful? | LDA `c_v` (K=10), reading-cross-checked with BERTopic | 0.55 |
| Do thematic profiles cluster by family? | Purity at k=7 in the chunk-level embedding space (chance = 0.14) | 0.54 |
| Are some families thematically more distinctive? | Cramér's V on the 7 × 29 family-by-topic contingency | 0.45 |
| Same topic, same tone? | Max range of family-mean sentiment on shared substantive topics | 0.25 on [-1, 1] |

The dominant pattern is an asymmetry between peripheral and mainstream families. Peripheral parties (FN, LO, écologistes, POE, PLN, PFN) each have at least one dedicated topic with a high lift; mainstream parties (PS, RPR, UDF) share a generic electoral register and concentrate in the HDBSCAN noise bucket (54% of chunks). The radical left is also the most critical voice on every shared topic — its mean sentiment is consistently 0.10+ below the corpus baseline (+0.63).

Side finding: some peripheral parties (Lutte Ouvrière in particular, but also the FN and the écologistes) distribute the same campaign tract to all their candidates, who just sign at the bottom. The within-topic ratio of distinct candidates to distinct chunk texts goes up to ~4 for these parties, vs ~1 for the mainstream. We didn't expect to see organisational signatures in a topic model, but they show up as flagged duplicate-chunks in the sanity digest.

### Hardware and runtime

Estimated runtime on Apple M-series with MPS acceleration (the configuration we used during development):

| Step | Approximate time |
|---|---|
| 01 → 03 | ~5 min |
| 04 (lemmatisation) | ~8 min |
| 06 (embedding ~118k chunks) | ~8 min |
| 07 (BERTopic + reduce_topics) | ~25 min |
| 08 (LDA, 10 passes) | ~8 min |
| 09 → 11 | ~5 min |
| 12 (sentiment, first run) | ~30 min (cached afterwards) |
| **Total** (cold full run) | ~1h30 |
