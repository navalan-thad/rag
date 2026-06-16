from typing import Dict, List, Any

class GraphRetriever:
    """
    Wraps a base retriever and uses a doc-level graph to expand candidates.

    Expects hits from base_retriever.retrieve(query, top_k) to include 'doc_id' and 'score'.
    """

    def __init__(
        self,
        base_retriever,
        doc_neighbors: Dict[str, List[str]],  # doc_id -> neighbor doc_ids
        chunks: List[Dict[str, Any]],
        neighbor_boost: float = 0.5,
        base_top_k: int = 20,
    ):
        self.base_retriever = base_retriever
        self.doc_neighbors = doc_neighbors
        self.chunks = chunks
        self.neighbor_boost = neighbor_boost
        self.base_top_k = base_top_k

        # Build lookup: doc_id -> chunks
        self._doc_to_chunks: Dict[str, List[Dict[str, Any]]] = {}
        for c in chunks:
            self._doc_to_chunks.setdefault(c["doc_id"], []).append(c)

    def retrieve(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        base_hits = self.base_retriever.retrieve(query, top_k=self.base_top_k)

        # Start with base scores at chunk level
        chunk_scores: Dict[str, float] = {}
        for h in base_hits:
            chunk_scores[h["chunk_id"]] = chunk_scores.get(h["chunk_id"], 0.0) + h["score"]

        # Expand via graph: for each hit's doc, add neighbors' chunks with discounted score
        for h in base_hits:
            doc_id = h["doc_id"]
            base_score = h["score"]
            for neighbor_doc in self.doc_neighbors.get(doc_id, []):
                neighbor_chunks = self._doc_to_chunks.get(neighbor_doc, [])
                for c in neighbor_chunks:
                    cid = c["chunk_id"]
                    chunk_scores[cid] = max(chunk_scores.get(cid, 0.0), base_score * self.neighbor_boost)

        # Turn scores back into hits list
        hits: List[Dict[str, Any]] = []
        for c in self.chunks:
            cid = c["chunk_id"]
            if cid in chunk_scores:
                hits.append({
                    "chunk_id": cid,
                    "doc_id": c["doc_id"],
                    "score": chunk_scores[cid],
                })

        hits.sort(key=lambda h: h["score"], reverse=True)
        return hits[:top_k]