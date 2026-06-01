from ingestion.loader import load_scifact
from ingestion.cleaner import build_clean_corpus
from chunking.chunkers import chunk_whole
from retrieval.embedder import Embedder
from retrieval.indexer import FaissIndex
from retrieval.retriever import DenseRetriever
from eval.metrics import evaluate, log_failures
import numpy as np

corpus, queries, qrels = load_scifact()
docs = build_clean_corpus(corpus)
chunks = chunk_whole(docs)

embedder = Embedder("all-MiniLM-L6-v2")
texts = [c["text"] for c in chunks]
embeddings = embedder.embed(texts)

index = FaissIndex(dim=embeddings.shape[1])
index.add(embeddings, [c["chunk_id"] for c in chunks])

retriever = DenseRetriever(embedder, index, chunks)

results = evaluate(retriever, queries, qrels, k_values=[1, 5, 10])
print(results)

failures = log_failures(retriever, queries, qrels, n=20)
for f in failures:
    print("\n--- FAILURE ---")
    print(f"Query:    {f['query']}")
    print(f"Relevant: {f['relevant_docs']}")
    print(f"Got:      {f['retrieved']}")
