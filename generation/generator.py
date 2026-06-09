from ollama import chat
from dataclasses import dataclass
from typing import List, Dict

RAG_PROMPT = """\
You are an AI research assistant. Answer the question using ONLY the provided context passages, which come from recent research papers.

Guidelines:
- Synthesize a concise answer (2–4 sentences) that cites the relevant passage indices in square brackets, e.g. [1], [2].
- If the passages are related but do not contain enough information to answer accurately, say:
  "Insufficient evidence in the retrieved papers." and briefly explain what is missing.
- Do NOT use any external knowledge beyond what appears in the context.

Context passages:
{context}

Question: {question}

Answer (with citations like [1], [2] referring to the passages above):
"""

@dataclass
class GenerationResult:
    question: str
    answer: str
    contexts: List[str]      # raw text of retrieved chunks
    source_ids: List[str]    # chunk_ids for traceability

def build_context(chunks: List[Dict], max_tokens: int = 1800) -> str:
    """Concatenate retrieved chunks into a numbered context block."""
    parts = []
    total = 0
    for i, chunk in enumerate(chunks):
        text = chunk["text"]
        word_estimate = len(text.split())
        if total + word_estimate > max_tokens:
            break
        parts.append(f"[{i+1}] {text}")
        total += word_estimate
    return "\n\n".join(parts)

def generate(
    question: str,
    retrieved_chunks: List[Dict],
    model: str = "llama3.2:3b",
    temperature: float = 0.0,
) -> GenerationResult:
    context = build_context(retrieved_chunks)
    prompt = RAG_PROMPT.format(context=context, question=question)

    response = chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": temperature},
    )
    answer = response.message.content.strip()

    return GenerationResult(
        question=question,
        answer=answer,
        contexts=[c["text"] for c in retrieved_chunks],
        source_ids=[c["chunk_id"] for c in retrieved_chunks],
    )