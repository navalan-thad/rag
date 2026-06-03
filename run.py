import config

from ingestion.loader import load_scifact
from ingestion.cleaner import build_clean_corpus
from chunking.chunkers import chunk_whole
from retrieval.embedder import Embedder
from retrieval.indexer import FaissIndex
from retrieval.retriever import DenseRetriever
from eval.metrics import evaluate
from eval.metrics import categorize_failures


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

failure_report = categorize_failures(retriever, queries, qrels, corpus, n=60)
for category, cases in failure_report.items():
    print(f"\n{category.upper()} ({len(cases)} cases)")
    for f in cases[:3]:
        print(f"  Query: {f['query']}")
        print(f"  Score: {f['top_score']:.3f}")
