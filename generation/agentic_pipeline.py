from typing import List
from generation.generator import generate, GenerationResult
from generation.query_rewriter import rewrite_question


class AgenticPipeline:
    """
    Wraps a base retriever + model with a simple agentic loop:
    1) retrieve with original query
    2) if confidence is low, rewrite query and re-retrieve
    3) generate answer from combined hits
    """

    def __init__(
        self,
        retriever,
        model: str = "llama3.2:3b",
        top_k: int = 5,
        rewrite_model: str = "llama3.2:3b",
        low_conf_threshold: float = 0.35,
    ):
        self.retriever = retriever
        self.model = model
        self.top_k = top_k
        self.rewrite_model = rewrite_model
        self.low_conf_threshold = low_conf_threshold

        # chunk_id -> full chunk dict
        self._chunk_lookup = {c["chunk_id"]: c for c in retriever.chunks}

    def _first_retrieve(self, question: str):
        hits = self.retriever.retrieve(question, self.top_k)
        if not hits:
            return hits, 0.0
        top_score = hits[0].get("score", 0.0)
        return hits, top_score

    def _retrieve_with_rewrite(self, question: str) -> List[dict]:
        # initial attempt
        hits, top_score = self._first_retrieve(question)
        print(f"[Agentic] original top_score={top_score:.3f}")  # ALWAYS logs

        if top_score >= self.low_conf_threshold:
            print("[Agentic] good enough, no rewrite")          # ALSO logs
            return hits

        print("[Agentic] low confidence, rewriting...")         # ONLY if rewriting
        notes = f"Top retrieval score was {top_score:.3f}, which is low."
        rewritten = rewrite_question(question, notes=notes, model=self.rewrite_model)
        print(f"[Agentic] rewritten: {rewritten}")

        rew_hits, rew_top = self._first_retrieve(rewritten)
        print(f"[Agentic] rewritten top_score={rew_top:.3f}")

        merged = {}
        for h in hits + rew_hits:
            cid = h["chunk_id"]
            merged[cid] = max(merged.get(cid, 0.0), h.get("score", 0.0))
        merged_hits = [
            {"chunk_id": cid, "doc_id": self._chunk_lookup[cid]["doc_id"], "score": s}
            for cid, s in merged.items()
        ]
        merged_hits.sort(key=lambda h: h["score"], reverse=True)
        return merged_hits[: self.top_k]

    def run(self, question: str) -> GenerationResult:
        hits = self._retrieve_with_rewrite(question)
        chunks = [
            self._chunk_lookup[h["chunk_id"]]
            for h in hits
            if h["chunk_id"] in self._chunk_lookup
        ]
        return generate(question, chunks, model=self.model)

    def run_batch(self, questions: List[str]) -> List[GenerationResult]:
        return [self.run(q) for q in questions]