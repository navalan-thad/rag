# RAG Research Project

A modular Retrieval-Augmented Generation (RAG) system built for experimentation and research on the SciFact biomedical corpus. The project is structured so each pipeline component can be swapped or ablated independently, with an automated experiment runner and live dashboard for tracking results.

---

## Project Structure

```
rag/
├── ingestion/
│   ├── loader.py          # Dataset loading (SciFact via BEIR)
│   └── cleaner.py         # Text normalization, title+body concatenation
├── chunking/
│   └── chunkers.py        # chunk_whole, chunk_fixed, chunk_sentences
├── retrieval/
│   ├── embedder.py        # SentenceTransformer wrapper with MPS/CPU support
│   ├── indexer.py         # FAISS index (IndexFlatIP, normalized vectors)
│   └── retriever.py       # DenseRetriever: query → ranked doc list
├── eval/
│   └── metrics.py         # Recall@k, MRR, log_failures
├── experiments/
│   ├── run_ablations.py   # Automated experiment runner
│   └── results.json       # Auto-generated results cache
├── app/
│   └── dashboard.html     # Live results dashboard
└── run.py                 # Quick single-run entrypoint
```

---

## Quickstart

### 1. Install dependencies

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the baseline

```bash
python run.py
```

This loads SciFact, embeds all passages, and prints Recall@1/5/10 and MRR to stdout.

### 3. Run all ablation experiments

```bash
python -m experiments.run_ablations
```

Results are saved to `experiments/results.json` as each config completes. If a run is interrupted, completed configs are skipped on re-run.

### 4. View the dashboard

```bash
python -m http.server 8000
```

Open `http://localhost:8000/app/dashboard.html`. The dashboard auto-refreshes every 30 seconds and updates as new results are written.

---

## Running on Apple Silicon (MPS)

MPS is used automatically when available. Two environment variables should be set before running:

```bash
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

`HIGH_WATERMARK_RATIO=0.0` removes PyTorch's artificial memory ceiling. `MPS_FALLBACK=1` routes unsupported ops to CPU silently. Add both to `~/.zshrc` to avoid setting them each session.

On 8 GB machines, reduce batch size for larger models:

```python
# In CONFIGS inside run_ablations.py
{"name": "pubmedbert", "model": "pritamdeka/S-PubMedBert-MS-MARCO", "batch_size": 16, ...}
```

---

## Adding a New Experiment

Open `experiments/run_ablations.py` and add an entry to the `CONFIGS` list:

```python
{"name": "my_experiment", "chunker": "whole", "model": "my-model-name", "hybrid": False, "rerank": False, "batch_size": 32}
```

Re-run `python -m experiments.run_ablations`. Existing results are cached — only the new config will execute.

---

## Tuning Knobs

### Parsing and Ingestion

Controlled in `ingestion/cleaner.py`.

The default strategy concatenates the document title and abstract body as a single string. Since SciFact titles are descriptive and semantically dense, this improves retrieval over body-only indexing. Alternatives to test:

- **Body only** — strip the title, useful if titles are generic or misleading.
- **Title-weighted repetition** — append the title twice to increase its influence on the embedding.
- **Metadata injection** — prepend structured fields such as year or journal before embedding.

Parsing quality is foundational. Errors introduced here propagate through every downstream component and cannot be recovered by better retrieval.

### Chunking

Controlled in `chunking/chunkers.py`. The `chunker` key in each config selects the strategy.

| Strategy | Description | When to use |
|---|---|---|
| `chunk_whole` | Each document is a single chunk | Short corpora like SciFact abstracts |
| `chunk_fixed` | Sliding window over word tokens | When passage length varies widely |
| `chunk_sentences` | Groups of N sentences | When sentence boundaries are clean |

Key parameters for `chunk_fixed`:

- `size` — number of tokens per chunk. Larger chunks improve recall but dilute precision.
- `overlap` — token overlap between adjacent chunks. Prevents answers from falling on boundaries.

For SciFact, `chunk_whole` performs well because passages are already short abstracts. On longer documents such as full papers, fixed or sentence chunking becomes necessary.

### Embedding Model

Controlled by the `model` key in each config entry. The model is the single highest-impact tuning knob in a dense retrieval system.

| Model | Parameters | Domain | Notes |
|---|---|---|---|
| `all-MiniLM-L6-v2` | 22M | General | Fast baseline, good for prototyping |
| `allenai-specter` | 110M | Scientific | Trained on citation pairs; strong on paper similarity |
| `multi-qa-mpnet-base-dot-v1` | 109M | General QA | Strong on question-passage matching tasks |
| `pritamdeka/S-PubMedBert-MS-MARCO` | 110M | Biomedical | Best observed performance on SciFact |

Domain alignment matters significantly. General models underperform on biomedical text because they have not seen the vocabulary and claim structures common in scientific literature.

### Retrieval Strategy

Controlled in `retrieval/retriever.py`. The `hybrid` flag in each config enables BM25 fusion.

- **Dense only** — semantic similarity via dot product on normalized embeddings. Misses exact-match queries involving rare terms, abbreviations, and specific numbers.
- **BM25 only** — lexical overlap via term frequency. Misses paraphrase and synonym relationships.
- **Hybrid (RRF)** — combines dense and BM25 rankings using Reciprocal Rank Fusion. Generally outperforms either alone, particularly on queries with domain-specific abbreviations.

Reciprocal Rank Fusion score for a document:

```
score(d) = 1 / (k + rank_dense(d)) + 1 / (k + rank_bm25(d))
```

where `k=60` is a smoothing constant that reduces the influence of very high-ranked documents.

### Reranking

Controlled by the `rerank` flag. A cross-encoder scores each (query, passage) pair jointly, which is more accurate than bi-encoder similarity but too slow for first-stage retrieval over a full corpus.

Recommended pattern: retrieve top 20 candidates with hybrid retrieval, then rerank to top 10.

```
Model: cross-encoder/ms-marco-MiniLM-L-6-v2
```

Reranking typically provides the largest single improvement after switching to a domain-appropriate embedding model.

### Top-k

The number of documents retrieved before reranking. Evaluated at k=1, 5, and 10. Increasing k improves Recall@k but increases prompt token usage and generation latency. For most QA tasks, k=5 to k=10 represents a practical trade-off.

### Prompt Construction

Not yet exposed as an ablation config. Variables to experiment with:

- **Context ordering** — whether the highest-scored passage appears first or last. Research on lost-in-the-middle suggests LLMs attend more to the beginning and end of context windows.
- **Citation format** — numbered references versus inline document IDs.
- **Abstention instruction** — whether the prompt instructs the model to refuse when evidence is absent. Critical for measuring hallucination rate.
- **Context truncation** — how to handle cases where top-k passages exceed the model's context window.

---

## Evaluation Metrics

Computed in `evaluation/metrics.py`.

| Metric | Description |
|---|---|
| `Recall@k` | Fraction of queries where at least one relevant document appears in the top k results |
| `MRR` | Mean Reciprocal Rank — average of 1/rank of the first relevant result across queries |
| `latency_s` | Wall-clock seconds for the evaluation loop over all test queries |

Retrieval metrics are computed against BEIR qrels, which provide binary relevance labels. Generation metrics (faithfulness, citation correctness, abstention accuracy) are not yet implemented and require an LLM judge or annotated answer set.

To inspect specific failures:

```python
from evaluation.metrics import log_failures
failures = log_failures(retriever, queries, qrels, n=20)
for f in failures:
    print(f["query"], f["retrieved"])
```

Common failure patterns observed on SciFact:

- **Semantic drift** — retrieved documents are topically adjacent but do not contain the specific claim. Most common with general-purpose embedding models.
- **Lexical gap** — query uses an abbreviation or specific term that dense embeddings compress away. BM25 hybrid addresses this directly.
- **Claim negation** — the query is a false claim to be verified; the relevant document refutes it. Dense retrieval conflates the claim with its negation.

---

## Extending the Project

### Adding a new chunker

Add a function to `chunking/chunkers.py` following the signature:

```python
def chunk_mysplit(docs) -> List[Dict]:
    # each item must have: chunk_id, doc_id, text
    ...
```

Register it in the `CHUNKER_MAP` in `run_ablations.py`:

```python
CHUNKER_MAP["mysplit"] = chunk_mysplit
```

### Adding a new retriever

Create a class in `retrieval/` with a `retrieve(query: str, top_k: int) -> List[Dict]` method where each result contains `doc_id` and `score`. Pass an instance to `evaluate()` in place of `DenseRetriever`.

### Adding a new dataset

Implement a loader in `ingestion/loader.py` that returns `(corpus, queries, qrels)` matching the BEIR format: `corpus` is a dict of `{doc_id: {title, text}}`, `queries` is `{query_id: query_text}`, and `qrels` is `{query_id: {doc_id: relevance_score}}`. No changes to the retrieval or evaluation modules are required.

### Switching to a larger corpus

Replace SciFact with BEIR/NQ for a harder retrieval task:

```python
from datasets import load_dataset
docs    = load_dataset('irds/beir_nq', 'docs')
queries = load_dataset('irds/beir_nq', 'queries')
qrels   = load_dataset('irds/beir_nq', 'qrels')
```

NQ contains 2.6 million Wikipedia passages. Shard to 50k passages for fast iteration before scaling to the full corpus.

---

## Requirements

```
numpy<2.0
torch>=2.4
transformers>=4.38,<4.50
sentence-transformers>=3.0
beir
faiss-cpu
datasets
rank_bm25
```

Install with `pip install -r requirements.txt`.
