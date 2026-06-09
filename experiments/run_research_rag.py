import typer
from typing import Optional

from ingestion.research_loader import load_arxiv_corpus
from ingestion.cleaner import build_clean_corpus
from chunking.chunkers import chunk_sentences

from retrieval.embedder import Embedder
from retrieval.indexer import FaissIndex
from retrieval.bm25_retriever import BM25Retriever
from retrieval.retriever import DenseRetriever, HybridRetriever
from retrieval.reranker import Reranker

from generation.rag_pipeline import RAGPipeline


app = typer.Typer(help="RAG over recent arXiv papers for a given topic.")


@app.command()
def run(
    topic: str = typer.Argument(..., help="Search query for arXiv, e.g. 'neural signal processing'"),
    n_papers: int = typer.Option(50, "--n-papers", help="Number of recent papers to pull"),
    model_name: str = typer.Option(
        "pritamdeka/S-PubMedBert-MS-MARCO",
        "--embedder-model",
        help="SentenceTransformers embedding model",
    ),
    use_hybrid: bool = typer.Option(True, "--hybrid/--no-hybrid", help="Use BM25 + dense hybrid?"),
    use_rerank: bool = typer.Option(True, "--rerank/--no-rerank", help="Use CrossEncoder reranker?"),
    top_k: int = typer.Option(8, "--top-k", help="Top-k passages to feed into generator"),
    llama_model: str = typer.Option("llama3.2:3b", "--llama-model", help="Ollama model name"),
):
    # 1. Load corpus from arXiv
    typer.echo(f"Loading ~{n_papers} recent papers on '{topic}' from arXiv...")
    corpus, _ = load_arxiv_corpus(topic, max_results=n_papers)

    # 2. Clean docs and chunk
    docs = build_clean_corpus(corpus)  # returns list of {"id": docid, "text": text}
    chunks = chunk_sentences(docs, max_sentences=5)

    # 3. Build dense index
    embedder = Embedder(model_name)  # handled inside Embedder
    texts = [c["text"] for c in chunks]
    embeddings = embedder.embed(texts)
    index = FaissIndex(dim=embeddings.shape[1])
    index.add(embeddings, [c["chunk_id"] for c in chunks])

    dense_retriever = DenseRetriever(embedder, index, chunks)

    # 4. Optional BM25 + hybrid
    if use_hybrid:
        bm25 = BM25Retriever(chunks)
        retriever = HybridRetriever(dense_retriever, bm25)
    else:
        retriever = dense_retriever

    # 5. Optional reranking
    if use_rerank:
        reranker = Reranker("cross-encoder/ms-marco-MiniLM-L-6-v2")
        base_retrieve = retriever.retrieve

        def retrieve_with_rerank(query: str, top_k: int = 10):
            candidates = base_retrieve(query, 20)
            return reranker.rerank(query, candidates, corpus, top_k)

        retriever.retrieve = retrieve_with_rerank

    # 6. RAG pipeline over this retriever
    pipeline = RAGPipeline(retriever=retriever, model=llama_model, top_k=top_k)

    # 7. Interactive loop
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
            typer.echo(f"[{i+1}] {ctx[:400].replace('\n', ' ')}...")
        typer.echo("-" * 80)


if __name__ == "__main__":
    app()