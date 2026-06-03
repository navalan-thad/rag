class DenseRetriever:
    def __init__(self, embedder, index, chunks):
        self.embedder = embedder
        self.index = index
        # map chunk_id → doc_id for evaluation
        self.chunk_to_doc = {c["chunk_id"]: c["doc_id"] for c in chunks}

    def retrieve(self, query: str, top_k=10):
        vec = self.embedder.embed([query], show_progress=False)
        hits = self.index.search(vec, top_k)
        for h in hits:
            h["doc_id"] = self.chunk_to_doc[h["chunk_id"]]
        return hits
    
    def reciprocal_rank_fusion(dense_hits, bm25_hits, k=60):
        scores = {}
        doc_lookup = {}

        for rank, hit in enumerate(dense_hits):
            doc_id = hit["doc_id"]
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
            doc_lookup[doc_id] = hit

        for rank, hit in enumerate(bm25_hits):
            doc_id = hit["doc_id"]
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
            doc_lookup[doc_id] = hit

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [{"doc_id": doc_id, "score": score} for doc_id, score in ranked]
    
class HybridRetriever:
    def __init__(self, dense_retriever, bm25_retriever):
        self.dense = dense_retriever
        self.bm25 = bm25_retriever

    def retrieve(self, query: str, top_k=10, first_stage_k=20):
        dense_hits = self.dense.retrieve(query, top_k=first_stage_k)
        bm25_hits = self.bm25.retrieve(query, top_k=first_stage_k)
        fused = DenseRetriever.reciprocal_rank_fusion(dense_hits, bm25_hits, k=first_stage_k)
        return fused[:top_k]