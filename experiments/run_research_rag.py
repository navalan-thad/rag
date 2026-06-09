import typer
from pipelines import build_arxiv_rag_pipeline

app = typer.Typer(help="RAG over recent arXiv papers for a given topic.")


@app.command()
def run(
    topic: str = typer.Argument(..., help="Search query for arXiv, e.g. 'neural signal processing'"),
    n_papers: int = typer.Option(50, "--n-papers", help="Number of recent papers to pull"),
    embedder_model: str = typer.Option(
        "pritamdeka/S-PubMedBert-MS-MARCO",
        "--embedder-model",
        help="SentenceTransformers embedding model",
    ),
    use_hybrid: bool = typer.Option(True, "--hybrid/--no-hybrid", help="Use BM25 + dense hybrid?"),
    use_rerank: bool = typer.Option(True, "--rerank/--no-rerank", help="Use CrossEncoder reranker?"),
    top_k: int = typer.Option(8, "--top-k", help="Top-k passages to feed into generator"),
    llama_model: str = typer.Option("llama3.2:3b", "--llama-model", help="Ollama model name"),
):
    pipeline, corpus = build_arxiv_rag_pipeline(
        topic=topic,
        n_papers=n_papers,
        embedder_model=embedder_model,
        use_hybrid=use_hybrid,
        use_rerank=use_rerank,
        top_k=top_k,
        llama_model=llama_model,
    )

    typer.echo("\nRAG ready. Ask questions about the downloaded papers.")
    typer.echo("Type 'exit' or Ctrl+C to quit.\n")

    while True:
        try:
            question = typer.prompt("Question")
        except (EOFError, KeyboardInterrupt):
            break
        if not question.strip() or question.strip().lower() in {"exit", "quit"}:
            break

        result = pipeline.run(question)
        typer.echo("\nAnswer:")
        typer.echo(result.answer)
        typer.echo("\nContexts used:")
        for i, ctx in enumerate(result.contexts):
            typer.echo(f"[{i+1}] {ctx[:400].replace('\\n', ' ')}...")
        typer.echo("-" * 80)


if __name__ == "__main__":
    app()