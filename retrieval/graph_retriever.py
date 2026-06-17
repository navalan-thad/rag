from typing import Dict, List, Any
from collections import defaultdict


class GraphRetriever:
    """
    Reranks using a doc-level k-NN graph:
    1) Get base dense hits.
    2) Build doc_scores from base hits.
    3) Propagate scores to neighbor docs.
    4) Assign doc_scores back to chunks and rank.
    """

    def __init__(
        self,
        base_retriever,
        doc_neighbors: Dict[str, List[str]],  # doc_id -> neighbor doc_ids
        chunks: List[Dict[str, Any]],
        neighbor_boost: float = 0.7,
        base_top_k: int = 50,
    ):
        self.base_retriever = base_retriever
        self.doc_neighbors = doc_neighbors
        self.chunks = chunks
        self.neighbor_boost = neighbor_boost
        self.base_top_k = base_top_k

        # doc_id -> chunks
        self._doc_to_chunks: Dict[str, List[Dict[str, Any]]] = {}
        for c in chunks:
            self._doc_to_chunks.setdefault(c["doc_id"], []).append(c)

    def retrieve(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        base_hits = self.base_retriever.retrieve(query, top_k=self.base_top_k)

        # 1) aggregate doc_scores from base hits
        doc_scores = defaultdict(float)
        for h in base_hits:
            doc_scores[h["doc_id"]] = max(doc_scores[h["doc_id"]], h["score"])

        # 2) propagate to neighbors
        propagated_scores = dict(doc_scores)
        for doc_id, base_score in doc_scores.items():
            for neighbor_doc in self.doc_neighbors.get(doc_id, []):
                propagated_scores[neighbor_doc] = max(
                    propagated_scores.get(neighbor_doc, 0.0),
                    base_score * self.neighbor_boost,
                )

        # 3) assign scores back to chunks
        chunk_scores: Dict[str, float] = {}
        for doc_id, score in propagated_scores.items():
            for c in self._doc_to_chunks.get(doc_id, []):
                cid = c["chunk_id"]
                chunk_scores[cid] = max(chunk_scores.get(cid, 0.0), score)

        # 4) build hits list
        hits: List[Dict[str, Any]] = []
        for c in self.chunks:
            cid = c["chunk_id"]
            if cid in chunk_scores:
                hits.append(
                    {"chunk_id": cid, "doc_id": c["doc_id"], "score": chunk_scores[cid]}
                )

        hits.sort(key=lambda h: h["score"], reverse=True)
        return hits[:top_k]