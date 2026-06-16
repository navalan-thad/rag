from typing import Dict, List
import numpy as np
from collections import defaultdict


def build_knn_graph(
    doc_embeddings: np.ndarray,
    doc_ids: List[str],
    k: int = 5,
) -> Dict[str, List[str]]:
    """
    Build a simple k-NN graph over documents based on cosine similarity.
    Returns adjacency: doc_id -> list of neighbor doc_ids.
    """
    # doc_embeddings shape: [N_docs, D], assumed L2-normalized
    sims = doc_embeddings @ doc_embeddings.T  # cosine since normalized
    np.fill_diagonal(sims, -1.0)  # avoid self

    neighbors = defaultdict(list)
    for i, doc_id in enumerate(doc_ids):
        top_indices = np.argpartition(sims[i], -k)[-k:]
        top_indices = top_indices[np.argsort(-sims[i, top_indices])]
        neighbors[doc_id] = [doc_ids[j] for j in top_indices]
    return neighbors