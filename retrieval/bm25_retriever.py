from rank_bm25 import BM25Okapi

class BM25Retriever:
    def __init__(self, chunks):
        self.chunk_ids = [c["chunk_id"] for c in chunks]
        self.chunk_to_doc = {c["chunk_id"]: c["doc_id"] for c in chunks}
        tokenized = [c["text"].lower().split() for c in chunks]
        self.bm25 = BM25Okapi(tokenized)

    def retrieve(self, query: str, top_k=20):
        scores = self.bm25.get_scores(query.lower().split())
        top_indices = scores.argsort()[::-1][:top_k]
        return [
            {
                "chunk_id": self.chunk_ids[i],
                "doc_id": self.chunk_to_doc[self.chunk_ids[i]],
                "score": float(scores[i])
            }
            for i in top_indices
        ]