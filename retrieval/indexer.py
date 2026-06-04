import faiss
import numpy as np

faiss.omp_set_num_threads(1)

class FaissIndex:
    def __init__(self, dim: int):
        self.index = faiss.IndexFlatIP(dim)  # inner product = cosine on normalized vecs
        self.id_map = []  # maps integer index → chunk_id

    def add(self, embeddings: np.ndarray, chunk_ids: list):
        self.index.add(embeddings.astype("float32"))
        self.id_map.extend(chunk_ids)

    def search(self, query_vec: np.ndarray, top_k=10):
        scores, indices = self.index.search(query_vec.astype("float32"), top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:
                results.append({"chunk_id": self.id_map[idx], "score": float(score)})
        return results