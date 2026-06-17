# RAG Lab: Retrieval-Augmented Generation Experiments on Scientific Text

This Retrieval-Augmented Generation project focuses on retrieval of scientific literature, with clean abstractions, rigorous evaluation, and several advanced retrieval variants (multi-modal, hybrid, graph-augmented, and agentic RAG).

---

## Goals and Scope

- Study how chunking, encoders, and retrieval strategies affect performance on a biomedical fact-checking benchmark (SciFact / BEIR).
- Build a robust RAG pipeline with explicit retrieval and end-to-end metrics, including RAGAS-based faithfulness.
- Explore frontier extensions: graph-augmented retrieval and agentic query rewriting, with hooks for future multimodal RAG.
- Provide reproducible scripts and a dashboard for running and comparing ablation experiments.

---

## High-Level Architecture

The codebase is structured as a set of reusable modules plus a small number of experiment scripts.

- **Ingestion**
  - `ingestion/loader.py`: downloads and loads the SciFact BEIR dataset into `corpus, queries, qrels`.
  - `ingestion/research_loader.py`: pulls recent arXiv papers for a given topic using the `arxiv` API and constructs a lightweight corpus.
  - `ingestion/cleaner.py`: normalizes docs into `{id, text}` (title + body) for downstream processing.

- **Chunking**
  - `chunking/chunkers.py`:
    - `chunk_whole`: one chunk per document.
    - `chunk_fixed`: fixed-size sliding windows over whitespace tokens.
    - `chunk_sentences`: sentence-group chunks (e.g., 5 sentences per chunk).
    - All chunkers output dicts with `chunk_id`, `doc_id`, `text` (and optional metadata like `image_path`).

- **Retrieval**
  - `retrieval/embedder.py`: wraps SentenceTransformers encoders (MiniLM, PubMedBERT, etc.) and runs on CPU/GPU/MPS.
  - `retrieval/indexer.py`: `FaissIndex` over L2-normalized embeddings for inner-product / cosine similarity.
  - `retrieval/retriever.py`:
    - `DenseRetriever`: dense vector search over chunks.
    - `HybridRetriever`: combines dense retrieval with BM25 via reciprocal rank fusion.
  - `retrieval/bm25_retriever.py`: lexical BM25 retrieval over chunks.
  - `retrieval/reranker.py`: CrossEncoder-based reranker for top candidate chunks.
  - `retrieval/graph_retriever.py`: graph-augmented retriever that reranks using a k-NN doc graph on top of dense retrieval.

- **Generation / RAG**
  - `generation/generator.py`:
    - Builds a numbered context block from retrieved chunks.
    - Sends a RAG prompt to an Ollama model (e.g. `llama3.2:3b`) with strict “answer using only context” instructions.
    - Returns a `GenerationResult` with `question`, `answer`, `contexts`, and `source_ids` (chunk_ids).
  - `generation/rag_pipeline.py`: `RAGPipeline` wrapping a retriever and model; exposes `run(question)` and `run_batch(questions)`.
  - `generation/agentic_pipeline.py`: `AgenticPipeline` that wraps a base retriever with a query-rewriting loop when confidence is low, then calls the same generator.

- **Evaluation**
  - `eval/metrics.py`:
    - `evaluate`: computes Recall@k and MRR over SciFact using qrels.
    - `categorize_failures`: tags per-query failures as lexical_gap, semantic_drift, low_confidence, or other based on scores and lexical overlap.
  - `eval/eval_e2e.py`:
    - `run_e2e_evaluation`: runs a RAG pipeline over a batch of questions and computes RAGAS metrics (faithfulness, etc.) using an Ollama judge model and embeddings.

- **Experiment Orchestration & Dashboard**
  - `experiments/run_ablations.py`: runs a suite of retrieval ablations (encoders, hybrid, rerank, chunking) over SciFact and writes results to `experiments/results.json` and `experiments/best_baseline.json`.
  - `experiments/run_full_rag.py`: uses the best retriever configuration to build a full RAG pipeline and run RAGAS end-to-end evaluation on SciFact.
  - `experiments/run_research_rag.py`: CLI research assistant over arXiv for a given topic, with optional hybrid and reranking, using a RAG pipeline in an interactive loop.
  - `app/dashboard.html`: a Chart.js-powered dashboard that reads `experiments/results.json` and shows a sortable table and bar chart of metrics across ablations.

- **Pipelines API**
  - `pipelines.py`: convenience builders such as:
    - `build_scifact_dense_retriever(...)`
    - `build_scifact_rag_pipeline(...)`
    - `build_scifact_graph_rag_pipeline(...)`
    - `build_scifact_agentic_rag_pipeline(...)`
    - `build_arxiv_rag_pipeline(...)`
    - All return either `(retriever, corpus, queries, qrels)` or `(pipeline, corpus, queries, qrels)` for direct use in scripts or notebooks.

---

## Capabilities

1. **Retrieval Benchmarking on SciFact (BEIR)**
   - Dense retrieval with multiple encoders (MiniLM, Specter, PubMedBERT, etc.).
   - BM25 retrieval and dense+BM25 hybrid via reciprocal rank fusion.
   - CrossEncoder reranking for top candidate chunks.
   - Chunking ablations (whole docs, fixed windows, sentence-based chunks).
   - Metrics: Recall@1/5/10 and MRR, plus basic failure analysis.

2. **End-to-End RAG Evaluation**
   - RAG pipeline with retrieval + LLM generation constrained to provided context.
   - RAGAS-based evaluation of answer faithfulness (and extension points for answer correctness).
   - Per-question outputs logging question, answer, source_ids, and ground-truth document text.

3. **Graph-Augmented Retrieval (Graph RAG)**
   - Document-level k-NN graph built over SciFact documents using pooled dense embeddings.
   - Graph-based reranking: propagate base dense scores across the document graph, then assign propagated scores back to chunks and re-rank.
   - Plug-and-play: `GraphRetriever` has the same interface as `DenseRetriever`, so it can be evaluated via the same metrics and used inside `RAGPipeline`.

4. **Agentic RAG (Query Rewriting Loop)**
   - `AgenticPipeline` wraps a base retriever and:
     - Runs initial retrieval and computes a confidence proxy (top score).
     - If low confidence, asks an LLM to rewrite the query to be clearer/more specific.
     - Re-runs retrieval on the rewritten query and merges original + rewritten hits before generation.
   - This allows experimentation with “self-reflection” style agentic RAG behaviors and measuring their impact on retrieval and RAGAS metrics.

5. **ArXiv Research Assistant**
   - CLI tool that:
     - Downloads recent arXiv papers for a topic (e.g. “neural signal processing”).
     - Cleans and sentence-chunks them.
     - Builds dense / hybrid / reranked retrieval over the temporary corpus.
     - Exposes an interactive QA loop answering questions with RAG over those papers, printing both answer and contexts.

---

## Tuning Knobs and Experiment Dimensions

The project is built around configurable “knobs” that can be systematically ablated.

- **Encoder model**
  - Choices in `run_ablations.py` / configs: `all-MiniLM-L6-v2`, `multi-qa-mpnet-base-dot-v1`, `allenai-specter`, `pritamdeka/S-PubMedBert-MS-MARCO`, etc.

- **Chunking**
  - `whole`: one chunk per document.
  - `fixed`: sliding windows with configurable `size` and `overlap`.
  - `sent`: sentence groups with configurable `max_sentences`.

- **Retrieval strategy**
  - Dense-only vs BM25-only vs hybrid (RRR + BM25).
  - With or without CrossEncoder reranking.
  - Graph-augmented vs standard dense/hybrid.

- **Batch size and hardware**
  - `Embedder` supports configurable batch sizes and automatically uses MPS/CPU/GPU based on availability.
  - `run_ablations.py` sets per-model batch sizes tuned for local performance.

- **RAG parameters**
  - `top_k`: number of passages fed into the generator.
  - LLM choice and temperature via Ollama (e.g., `llama3.2:3b`).

- **Agentic settings**
  - `low_conf_threshold`: controls when query rewriting triggers.
  - Rewriter model (can reuse the same LLM as the generator or use a lighter model).

- **Graph settings**
  - k for k-NN document graph.
  - `neighbor_boost`: how strongly neighbor docs are promoted in scoring.
  - `base_top_k`: number of base dense hits considered before graph propagation.

These knobs are primarily configured in `experiments/run_ablations.py`, `pipelines.py`, and the experiment scripts; they can also be moved into JSON/YAML configs for reproducibility.

---

## How to Run and Reproduce Experiments

Assuming Python 3.10+ and a working FAISS / PyTorch / SentenceTransformers setup, plus Ollama running with the required models pulled.

### 1. Install dependencies

```bash
pip install -r requirements.txt
# or equivalent environment.yml / pip install -e .
```

Models used include SentenceTransformers encoders, BEIR, RAGAS, arxiv, FAISS, and Ollama-related clients.

### 2. SciFact retrieval baseline

```bash
python run.py
```

This:

- Downloads SciFact via BEIR (if not cached).
- Builds a dense retriever over whole-doc chunks using `all-MiniLM-L6-v2`.
- Prints Recall@1/5/10 and MRR.
- Logs a categorized failure report for a sample of queries.

### 3. Retrieval ablations + dashboard

```bash
python experiments/run_ablations.py
```

This:

- Runs multiple configurations (encoders, hybrid vs dense, rerank vs no rerank, chunking variants).
- Appends results to `experiments/results.json` and updates `experiments/best_baseline.json`.

To view:

```bash
# For a simple static server (Python 3)
cd app
python -m http.server 8000
# then open http://localhost:8000/dashboard.html
```

The dashboard shows:

- A sortable table of experiments with Recall@1/5/10, MRR, and latency.
- A bar chart comparing MRR and Recall@10 across configurations.
- Highlights the best MRR configuration.

### 4. End-to-end RAG + RAGAS on SciFact

```bash
python experiments/run_full_rag.py
```

This:

- Uses the best retrieval config (e.g. sentence chunks + PubMedBERT + hybrid + rerank) to build a `RAGPipeline`.
- Samples queries with at least one relevant document.
- Builds ground-truth reference answers from the SciFact corpus.
- Runs RAG over those queries and evaluates with RAGAS (faithfulness), writing `results/e2e_eval.json`.

### 5. Graph RAG experiments

- To compare dense vs graph-augmented retrieval:

  ```bash
  python experiments/run_graph_ablations.py
  ```

  This script (or a similar one) builds `GraphRetriever` on top of the best dense retriever, runs `eval.metrics.evaluate`, and prints metrics.

- To run full RAG with graph-augmented retrieval:

  ```bash
  python experiments/run_graph_full_rag.py
  ```

  This uses `build_scifact_graph_rag_pipeline` and `run_e2e_evaluation` to produce a `results/e2e_eval_graph.json` comparable to the baseline.

### 6. Agentic RAG experiments

- Retrieval+generation for a sample of hard queries (via a debugging script) to inspect when query rewriting triggers.
- Full e2e RAG + RAGAS comparison:

  ```bash
  python experiments/run_agentic_full_rag.py
  ```

  This uses `build_scifact_agentic_rag_pipeline` and writes `results/e2e_eval_agentic.json`.

### 7. ArXiv research assistant

```bash
python -m experiments.run_research_rag "neural signal processing" \
  --n-papers 50 \
  --embedder-model "pritamdeka/S-PubMedBert-MS-MARCO" \
  --hybrid \
  --rerank \
  --top-k 8 \
  --llama-model "llama3.2:3b"
```

This starts an interactive loop:

- Downloads recent arXiv papers matching the topic.
- Builds a retrieval index.
- Answers free-form questions using RAG over the downloaded corpus, printing both answers and context snippets.

