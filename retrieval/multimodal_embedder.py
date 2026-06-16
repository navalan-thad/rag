from typing import List, Dict, Optional
import numpy as np
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import CLIPModel, CLIPProcessor


class MultimodalEmbedder:
    """
    Embed text and images (optional) into a single vector per chunk.
    Compatible with existing FaissIndex
    """

    def __init__(
        self,
        text_model_name: str = "pritamdeka/S-PubMedBert-MS-MARCO",
        clip_model_name: str = "openai/clip-vit-base-patch32",
        device: Optional[str] = None,
        alpha: float = 0.7,  # weight for text vs image
    ):
        self.text_model = SentenceTransformer(text_model_name)
        self.text_model.max_seq_length = 512

        self.clip_model = CLIPModel.from_pretrained(clip_model_name)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)

        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self.device = device
        self.clip_model.to(self.device)

        self.alpha = alpha

    def _embed_text(self, texts: List[str]) -> np.ndarray:
        return self.text_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

    def _embed_images(self, image_paths: List[Optional[str]]) -> np.ndarray:
        # Return zero vectors when no image is present
        images = []
        idx_map = []
        for i, path in enumerate(image_paths):
            if path:
                images.append(Image.open(path).convert("RGB"))
                idx_map.append(i)

        if not images:
            print("No images found in chunks; returning zero vectors for all.")
            return np.zeros((len(image_paths), self.clip_model.config.projection_dim), dtype=np.float32)

        inputs = self.clip_processor(images=images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.clip_model.get_image_features(**inputs)
        outputs = outputs / outputs.norm(dim=-1, keepdim=True)
        img_emb = outputs.cpu().numpy()

        # Scatter back into full array
        full = np.zeros((len(image_paths), img_emb.shape[1]), dtype=np.float32)
        for row_idx, emb_idx in zip(idx_map, range(len(img_emb))):
            full[row_idx] = img_emb[emb_idx]
        return full

    def embed_chunks(self, chunks: List[Dict]) -> np.ndarray:
        texts = [c["text"] for c in chunks]
        image_paths = [c.get("image_path") for c in chunks]

        text_emb = self._embed_text(texts)
        img_emb = self._embed_images(image_paths)

        # Simple weighted sum fusion; both are L2-normalized beforehand ideally.
        fused = self.alpha * text_emb + (1.0 - self.alpha) * img_emb

        # Normalize to unit length for cosine via inner product
        norms = np.linalg.norm(fused, axis=1, keepdims=True) + 1e-12
        fused = fused / norms
        return fused