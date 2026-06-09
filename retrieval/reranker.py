from sentence_transformers import CrossEncoder

class Reranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, hits: list, corpus: dict, top_k=10):
        # hits: list of dicts, must contain at least "doc_id"
        pairs = [(query, corpus[h["doc_id"]]["text"]) for h in hits]
        scores = self.model.predict(pairs)
        reranked = sorted(
            zip(hits, scores),
            key=lambda x: x[1],
            reverse=True
        )
        out = []
        for h, s in reranked[:top_k]:
            h2 = dict(h)              # copy ALL keys (chunk_id, doc_id, score, etc.)
            h2["score"] = float(s)    # overwrite score with reranked score
            out.append(h2)
        return out