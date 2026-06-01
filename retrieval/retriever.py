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