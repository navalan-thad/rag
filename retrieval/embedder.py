import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

from sentence_transformers import SentenceTransformer
import torch

class Embedder:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        if torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        self.model = SentenceTransformer(model_name, device=self.device)
        print(f"Embedder using: {self.device}")

    def embed(self, texts: list, batch_size=32, show_progress=True):
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
            convert_to_numpy=True
        )