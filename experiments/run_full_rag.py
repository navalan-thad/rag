from typing import Dict, List, Tuple
import random

from eval.eval_e2e import run_e2e_evaluation
from pipelines import build_scifact_rag_pipeline


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

    query_ids: List[str] = []
    query_texts: List[str] = []
    ground_truths: List[str] = []

    for qid in sampled_qids:
        # pick the most relevant doc as the ground truth reference
        best_doc_id = max(qrels[qid], key=lambda did: qrels[qid][did])
        doc = corpus[best_doc_id]
        ground_truth = f"{doc.get('title', '')} {doc['text']}".strip()

        query_ids.append(qid)
        query_texts.append(queries[qid])
        ground_truths.append(ground_truth)

    return query_ids, query_texts, ground_truths


def main() -> None:
    # Build the best SciFact RAG pipeline 
    pipeline, corpus, queries, qrels = build_scifact_rag_pipeline(
        model_name="pritamdeka/PubMedBERT-mnli-snli-scinli-scitail-mednli-stsb",
        chunker="sent",
        llama_model="llama3.2:3b",
        top_k=5,
    )

    # Build e2e evaluation inputs from existing qrels and corpus
    query_ids, query_texts, ground_truths = build_e2e_inputs(
        queries, qrels, corpus, n_samples=10
    )

    print(f"Running e2e evaluation on {len(query_texts)} queries...")
    run_e2e_evaluation(pipeline, query_texts, ground_truths, "results/e2e_eval.json")


if __name__ == "__main__":
    main()