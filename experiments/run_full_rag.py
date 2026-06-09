# IMPORT scifact
from ingestion.loader import load_scifact
from ingestion.cleaner import build_clean_corpus
from retrieval.embedder import Embedder
from retrieval.indexer import FaissIndex
from retrieval.retriever import DenseRetriever
from generation.rag_pipeline import RAGPipeline
from eval.eval_e2e import run_e2e_evaluation
from typing import Dict, List, Tuple
import random
from .run_ablations import CHUNKER_MAP


def build_e2e_inputs(
    queries: Dict[str, str],
    qrels: Dict[str, Dict[str, int]],
    corpus: Dict[str, Dict],
    n_samples: int = 10,
    seed: int = 42,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Returns three aligned lists:
      query_ids      - for logging
      query_texts    - fed to the pipeline
      ground_truths  - the gold document text used by RAGAS as reference answer
    
    Selects queries that have at least one relevant document (relevance >= 1).
    Uses the highest-scored relevant document as ground truth.
    """
    random.seed(seed)

    eligible = [
        qid for qid, rels in qrels.items()
        if any(score >= 1 for score in rels.values())
    ]
    sampled_qids = random.sample(eligible, min(n_samples, len(eligible)))

    query_ids, query_texts, ground_truths = [], [], []

    for qid in sampled_qids:
        # pick the most relevant doc as the ground truth reference
        best_doc_id = max(qrels[qid], key=lambda did: qrels[qid][did])
        doc = corpus[best_doc_id]
        ground_truth = f"{doc.get('title', '')} {doc['text']}".strip()

        query_ids.append(qid)
        query_texts.append(queries[qid])
        ground_truths.append(ground_truth)

    return query_ids, query_texts, ground_truths

corpus, queries, qrels = load_scifact()

# best retriever from ablation experiments
chunker_name = "sent"
docs = build_clean_corpus(corpus)
chunks = CHUNKER_MAP[chunker_name](docs)

embedder = Embedder("pritamdeka/PubMedBERT-mnli-snli-scinli-scitail-mednli-stsb")
embeddings = embedder.embed([c["text"] for c in chunks])
index = FaissIndex(dim=embeddings.shape[1])
index.add(embeddings, [c["chunk_id"] for c in chunks])
retriever = DenseRetriever(embedder, index, chunks)

# wiring pipeline
pipeline = RAGPipeline(retriever=retriever, model="llama3.2:3b", top_k=5)

# building inputs from existing qrels and corpus
query_ids, query_texts, ground_truths = build_e2e_inputs(
    queries, qrels, corpus, n_samples=10
)

print(f"Running e2e evaluation on {len(query_texts)} queries...")
run_e2e_evaluation(pipeline, query_texts, ground_truths, "results/e2e_eval.json")