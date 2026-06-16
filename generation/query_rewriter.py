from ollama import chat

REWRITE_PROMPT = """\
You rewrite search queries to make them clearer and more specific.
Return only the rewritten query as a single line, with no explanation.

Original question:
{question}

Optional notes about retrieval issues:
{notes}

Rewrite the question into a single, precise query that will retrieve the most relevant scientific papers.
Do not answer the question; only rewrite it.
"""

def rewrite_question(
    question: str,
    notes: str = "",
    model: str = "llama3.2:3b",
    temperature: float = 0.0,
) -> str:
    prompt = REWRITE_PROMPT.format(question=question, notes=notes)
    response = chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": temperature},
    )
    return response.message.content.strip()