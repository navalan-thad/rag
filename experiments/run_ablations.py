import config
import gc
import torch
import json, time
from pathlib import Path
from ingestion.loader import load_scifact
from ingestion.cleaner import build_clean_corpus
from chunking.chunkers import chunk_whole, chunk_fixed, chunk_sentences
from retrieval.embedder import Embedder
from retrieval.indexer import FaissIndex
from retrieval.retriever import DenseRetriever
from eval.metrics import evaluate

from retrieval.bm25_retriever import BM25Retriever
from retrieval.retriever import DenseRetriever, HybridRetriever
from retrieval.reranker import Reranker

if torch.backends.mps.is_available():
    torch.mps.empty_cache()
gc.collect()

CONFIGS = [
    # embedding model ablations (dense only)
    {"name": "baseline",        "chunker": "whole", "model": "all-MiniLM-L6-v2",                   "hybrid": False, "rerank": False, "batch_size": 64},
    {"name": "specter",         "chunker": "whole", "model": "allenai-specter",                     "hybrid": False, "rerank": False, "batch_size": 32},
    {"name": "multi_qa_mpnet",  "chunker": "whole", "model": "multi-qa-mpnet-base-dot-v1",          "hybrid": False, "rerank": False, "batch_size": 32},
    {"name": "pubmedbert",      "chunker": "whole", "model": "pritamdeka/S-PubMedBert-MS-MARCO",    "hybrid": False, "rerank": False, "batch_size": 16},

    # retrieval strategy ablations (best model)
    {"name": "hybrid_rrf",      "chunker": "whole", "model": "pritamdeka/S-PubMedBert-MS-MARCO",    "hybrid": True,  "rerank": False, "batch_size": 16},
    {"name": "hybrid_rerank",   "chunker": "whole", "model": "pritamdeka/S-PubMedBert-MS-MARCO",    "hybrid": True,  "rerank": True,  "batch_size": 16},

    # chunking ablations (best model + best retrieval)
    {"name": "chunk_fixed_200", "chunker": "fixed", "model": "pritamdeka/S-PubMedBert-MS-MARCO",    "hybrid": True,  "rerank": True,  "batch_size": 16},
    {"name": "chunk_sent_5",    "chunker": "sent",  "model": "pritamdeka/S-PubMedBert-MS-MARCO",    "hybrid": True,  "rerank": True,  "batch_size": 16},
]

CHUNKER_MAP = {
    "whole": chunk_whole,
    "fixed": lambda docs: chunk_fixed(docs, size=200, overlap=20),
    "sent":  lambda docs: chunk_sentences(docs, max_sentences=5),
}

RESULTS_FILE = Path("experiments/results.json")

def load_results():
    if RESULTS_FILE.exists():
        return json.loads(RESULTS_FILE.read_text())
    return []

def save_result(entry):
    results = load_results()
    # overwrite if same name exists
    results = [r for r in results if r["name"] != entry["name"]]
    results.append(entry)
    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    print(f"  Saved → {entry['name']}: {entry['metrics']}")

def run_config(cfg, corpus, queries, qrels):
    docs = build_clean_corpus(corpus)
    chunks = CHUNKER_MAP[cfg["chunker"]](docs)

    embedder = Embedder(cfg["model"])
    texts = [c["text"] for c in chunks]
    embeddings = embedder.embed(texts, batch_size=cfg.get("batch_size", 32))

    index = FaissIndex(dim=embeddings.shape[1])
    index.add(embeddings, [c["chunk_id"] for c in chunks])
    dense_retriever = DenseRetriever(embedder, index, chunks)

    if cfg.get("hybrid"):
        bm25_retriever = BM25Retriever(chunks)
        retriever = HybridRetriever(dense_retriever, bm25_retriever)
    else:
        retriever = dense_retriever

    if cfg.get("rerank"):
        reranker = Reranker()
        # wrap retrieve to add reranking transparently
        base_retrieve = retriever.retrieve
        def retrieve_with_rerank(query, top_k=10):
            candidates = base_retrieve(query, top_k=20)
            return reranker.rerank(query, candidates, corpus, top_k=top_k)
        retriever.retrieve = retrieve_with_rerank

    t0 = time.time()
    metrics = evaluate(retriever, queries, qrels, k_values=[1, 5, 10])
    metrics["latency_s"] = round(time.time() - t0, 2)

    del embedder, index, dense_retriever, embeddings
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return metrics

def get_best_baseline(metric="MRR"):
    results = json.loads(Path("experiments/results.json").read_text())
    return max(results, key=lambda r: r["metrics"].get(metric, 0))

if __name__ == "__main__":
    corpus, queries, qrels = load_scifact()
    existing = {r["name"] for r in load_results()}

    for cfg in CONFIGS:
        if cfg["name"] in existing:
            print(f"  Skipping {cfg['name']} (already run)")
            continue
        print(f"\nRunning: {cfg['name']} ...")
        metrics = run_config(cfg, corpus, queries, qrels)
        save_result({"name": cfg["name"], "config": cfg, "metrics": metrics})

        best = get_best_baseline()
        print(f"\nBest config: {best['name']}")
        print(f"Metrics: {best['metrics']}")
        Path("experiments/best_baseline.json").write_text(json.dumps(best, indent=2))