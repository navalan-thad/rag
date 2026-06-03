from sentence_transformers import CrossEncoder

class Reranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, hits: list, corpus: dict, top_k=10):
        # hits: list of {"doc_id": ..., "score":..}
        pairs = [(query, corpus[h["doc_id"]]["text"]) for h in hits]
        scores = self.model.predict(pairs)
        reranked = sorted(
            zip(hits, scores),
            key=lambda x: x[1],
            reverse=True
        )
        return [
            {"doc_id": h["doc_id"], "score": float(s)}
            for h, s in reranked[:top_k]
        ]