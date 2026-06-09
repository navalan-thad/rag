from generation.generator import generate, GenerationResult
from typing import List, Dict, Any


class RAGPipeline:
    def __init__(self, retriever, model: str = "llama3.2:3b", top_k: int = 5):
        self.retriever = retriever
        self.model = model
        self.top_k = top_k
        # chunk_id -> full chunk dict
        self._chunk_lookup: Dict[str, Dict[str, Any]] = {
            c["chunk_id"]: c for c in retriever.chunks
        }

    def run(self, question: str) -> GenerationResult:
        hits = self.retriever.retrieve(question, self.top_k)
        chunks = [
            self._chunk_lookup[h["chunk_id"]]
            for h in hits
            if h["chunk_id"] in self._chunk_lookup
        ]
        return generate(question, chunks, model=self.model)

    def run_batch(self, questions: List[str]) -> List[GenerationResult]:
        return [self.run(q) for q in questions]