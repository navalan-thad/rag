class DenseRetriever:
    def __init__(self, embedder, index, chunks):
        self.embedder = embedder
        self.index = index
        self.chunks = chunks
        self.chunk_to_doc = {c["chunk_id"]: c["doc_id"] for c in chunks}

    def retrieve(self, query: str, top_k=10):
        vec = self.embedder.embed([query], show_progress=False)
        hits = self.index.search(vec, top_k)
        # hits: [{"chunk_id": ..., "score": ...}]
        for h in hits:
            h["doc_id"] = self.chunk_to_doc[h["chunk_id"]]
        return hits

    def reciprocal_rank_fusion(dense_hits, bm25_hits, k=60):
        """
        Fuses at the CHUNK level, but keeps doc_id for metrics.

        dense_hits and bm25_hits are lists of:
          {"chunk_id": ..., "doc_id": ..., "score": ...}
        """
        scores = {}
        hit_lookup = {}

        # dense side
        for rank, hit in enumerate(dense_hits):
            cid = hit["chunk_id"]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
            hit_lookup[cid] = hit

        # bm25 side
        for rank, hit in enumerate(bm25_hits):
            cid = hit["chunk_id"]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
            hit_lookup[cid] = hit

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        # keep BOTH chunk_id and doc_id
        return [
            {
                "chunk_id": cid,
                "doc_id": hit_lookup[cid]["doc_id"],
                "score": score,
            }
            for cid, score in ranked
        ]


class HybridRetriever:
    def __init__(self, dense_retriever, bm25_retriever):
        self.dense = dense_retriever
        self.bm25 = bm25_retriever
        # expose chunks for RAGPipeline
        self.chunks = dense_retriever.chunks

    def retrieve(self, query: str, top_k=10, first_stage_k=20):
        dense_hits = self.dense.retrieve(query, top_k=first_stage_k)
        bm25_hits = self.bm25.retrieve(query, top_k=first_stage_k)
        fused = DenseRetriever.reciprocal_rank_fusion(
            dense_hits, bm25_hits, k=first_stage_k
        )
        return fused[:top_k]