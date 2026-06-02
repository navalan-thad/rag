# force cpu
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

import torch, gc
import json, time, itertools
from pathlib import Path
from ingestion.loader import load_scifact
from ingestion.cleaner import build_clean_corpus
from chunking.chunkers import chunk_whole, chunk_fixed, chunk_sentences
from retrieval.embedder import Embedder
from retrieval.indexer import FaissIndex
from retrieval.retriever import DenseRetriever
from eval.metrics import evaluate

if torch.backends.mps.is_available():
    torch.mps.empty_cache()
gc.collect()

CONFIGS = [
    {"name": "baseline",         "chunker": "whole", "model": "all-MiniLM-L6-v2",                     "batch_size": 64, "hybrid": False, "rerank": False},
    {"name": "specter",          "chunker": "whole", "model": "allenai-specter",                      "batch_size": 32, "hybrid": False, "rerank": False},
    {"name": "mutli_qa_mpnet",   "chunker": "whole", "model": "multi-qa-mpnet-base-dot-v1",           "batch_size": 32, "hybrid": False, "rerank": False},
    {"name": "pubmedbert",       "chunker": "whole", "model": "pritamdeka/S-PubMedBert-MS-MARCO",     "batch_size": 16, "hybrid": False, "rerank": False},
    # {"name": "hybrid_rrf",       "chunker": "whole", "model": "pritamdeka/S-PubMedBert-MS-MARCO",     "batch_size": 16, "hybrid": True,  "rerank": False},
    # {"name": "hybrid_rerank",    "chunker": "whole", "model": "pritamdeka/S-PubMedBert-MS-MARCO",     "batch_size": 16, "hybrid": True,  "rerank": True},
    # {"name": "chunk_fixed_200",  "chunker": "fixed", "model": "pritamdeka/S-PubMedBert-MS-MARCO",     "batch_size": 16, "hybrid": True,  "rerank": True},
    # {"name": "chunk_sent_5",     "chunker": "sent",  "model": "pritamdeka/S-PubMedBert-MS-MARCO",     "batch_size": 16, "hybrid": True,  "rerank": True},
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
    embeddings = embedder.embed(texts)

    index = FaissIndex(dim=embeddings.shape[1])
    index.add(embeddings, [c["chunk_id"] for c in chunks])
    retriever = DenseRetriever(embedder, index, chunks)

    t0 = time.time()
    metrics = evaluate(retriever, queries, qrels, k_values=[1, 5, 10])
    metrics["latency_s"] = round(time.time() - t0, 2)

    # cleanup
    del embedder, index, retriever, embeddings
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
        
    return metrics

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
    
