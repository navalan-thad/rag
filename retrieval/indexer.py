import faiss
import numpy as np

class FaissIndex:
    def __init__(self, dim: int):
        # inner product on normalized vectors = cosine similarity
        self.index = faiss.IndexFlatIP(dim)
        self.idmap: list[str] = []  # integer -> chunk_id

    def add(self, embeddings: np.ndarray, chunk_ids: list[str]) -> None:
        self.index.add(embeddings.astype("float32"))
        self.idmap.extend(chunk_ids)

    def search(self, query_vec: np.ndarray, top_k: int = 10):
        if query_vec.ndim == 1:
            query_vec = query_vec[None, :]
        scores, indices = self.index.search(query_vec.astype("float32"), top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            cid = self.idmap[idx]
            results.append({
                "chunk_id": cid,
                "score": float(score),
            })
        return results