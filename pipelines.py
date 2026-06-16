from typing import Dict, Tuple
import numpy as np
from collections import defaultdict
from ingestion.loader import load_scifact
from ingestion.cleaner import build_clean_corpus
from ingestion.research_loader import load_arxiv_corpus

from chunking.chunkers import chunk_whole, chunk_fixed, chunk_sentences
from retrieval.embedder import Embedder
from retrieval.indexer import FaissIndex
from retrieval.retriever import DenseRetriever, HybridRetriever
from retrieval.bm25_retriever import BM25Retriever
from retrieval.reranker import Reranker
from generation.rag_pipeline import RAGPipeline
from retrieval.multimodal_embedder import MultimodalEmbedder

from retrieval.graph_retriever import GraphRetriever
from graph.graph_builder import build_knn_graph


def build_scifact_dense_retriever(
    model_name: str = "all-MiniLM-L6-v2",
    chunker: str = "whole",
):
    """
    Build a dense retriever over SciFact for retrieval-only experiments.
    Returns (retriever, corpus, queries, qrels).
    """
    corpus, queries, qrels = load_scifact()
    docs = build_clean_corpus(corpus)

    if chunker == "whole":
        chunks = chunk_whole(docs)
    elif chunker == "fixed":
        chunks = chunk_fixed(docs, size=200, overlap=20)
    elif chunker == "sent":
        chunks = chunk_sentences(docs, max_sentences=5)
    else:
        raise ValueError(f"Unknown chunker: {chunker}")

    embedder = Embedder(model_name)
    texts = [c["text"] for c in chunks]
    embeddings = embedder.embed(texts)

    index = FaissIndex(dim=embeddings.shape[1])
    index.add(embeddings, [c["chunk_id"] for c in chunks])

    retriever = DenseRetriever(embedder, index, chunks)
    return retriever, corpus, queries, qrels


def build_scifact_rag_pipeline(
    model_name: str = "pritamdeka/PubMedBERT-MS-MARCO",
    chunker: str = "sent",
    llama_model: str = "llama3.2:3b",
    top_k: int = 5,
) -> Tuple[RAGPipeline, dict, dict, dict]:
    """
    Build a full RAG pipeline over SciFact using the best-known config.
    Returns (pipeline, corpus, queries, qrels).
    """
    retriever, corpus, queries, qrels = build_scifact_dense_retriever(
        model_name=model_name,
        chunker=chunker,
    )
    pipeline = RAGPipeline(retriever=retriever, model=llama_model, top_k=top_k)
    return pipeline, corpus, queries, qrels


def build_arxiv_rag_pipeline(
    topic: str,
    n_papers: int = 50,
    embedder_model: str = "pritamdeka/S-PubMedBert-MS-MARCO",
    use_hybrid: bool = True,
    use_rerank: bool = True,
    top_k: int = 8,
    llama_model: str = "llama3.2:3b",
) -> Tuple[RAGPipeline, dict]:
    """
    Build a RAG pipeline over recent arXiv papers for a given topic.
    Returns (pipeline, corpus).
    """
    corpus, _ = load_arxiv_corpus(topic, max_results=n_papers)
    docs = build_clean_corpus(corpus)
    chunks = chunk_sentences(docs, max_sentences=5)

    embedder = Embedder(embedder_model)
    texts = [c["text"] for c in chunks]
    embeddings = embedder.embed(texts)
    index = FaissIndex(dim=embeddings.shape[1])
    index.add(embeddings, [c["chunk_id"] for c in chunks])

    dense_retriever = DenseRetriever(embedder, index, chunks)

    if use_hybrid:
        bm25 = BM25Retriever(chunks)
        retriever = HybridRetriever(dense_retriever, bm25)
    else:
        retriever = dense_retriever

    if use_rerank:
        reranker = Reranker("cross-encoder/ms-marco-MiniLM-L-6-v2")
        base_retrieve = retriever.retrieve

        def retrieve_with_rerank(query: str, top_k_inner: int = 10):
            candidates = base_retrieve(query, 20)
            return reranker.rerank(query, candidates, corpus, top_k_inner)

        retriever.retrieve = retrieve_with_rerank

    pipeline = RAGPipeline(retriever=retriever, model=llama_model, top_k=top_k)
    return pipeline, corpus

def build_multimodal_rag_pipeline(
    corpus: Dict[str, Dict],
    llama_model: str = "llama3.2:3b",
    top_k: int = 5,
) -> RAGPipeline:
    """
    Example: build a RAG pipeline over a corpus where docs may have images.
    """
    docs = build_clean_corpus(corpus)  # should include image_path if available
    chunks = chunk_sentences(docs, max_sentences=5)

    embedder = MultimodalEmbedder()
    embeddings = embedder.embed_chunks(chunks)

    index = FaissIndex(dim=embeddings.shape[1])
    index.add(embeddings, [c["chunk_id"] for c in chunks])

    retriever = DenseRetriever(embedder=None, index=index, chunks=chunks)  # embedder not used at query time in this simple version

    pipeline = RAGPipeline(retriever=retriever, model=llama_model, top_k=top_k)
    return pipeline

def build_scifact_graph_rag_pipeline(
    model_name: str = "pritamdeka/S-PubMedBert-MS-MARCO",
    chunker: str = "sent",
    llama_model: str = "llama3.2:3b",
    top_k: int = 5,
) -> Tuple[RAGPipeline, dict, dict, dict]:
    """
    SciFact RAG pipeline with graph-augmented retrieval.
    """
    # Reuse existing helper to get a dense retriever
    base_retriever, corpus, queries, qrels = build_scifact_dense_retriever(
        model_name=model_name,
        chunker=chunker,
    )
    chunks = base_retriever.chunks
    # Assume embeddings were computed when building the retriever, otherwise recompute from texts
    embedder = Embedder(model_name)
    texts = [c["text"] for c in chunks]
    chunk_emb = embedder.embed(texts)  # shape [N_chunks, D], assumed normalized

    # Aggregate by doc_id
    doc_vecs = defaultdict(list)
    for c, emb in zip(chunks, chunk_emb):
        doc_vecs[c["doc_id"]].append(emb)
    doc_ids = list(doc_vecs.keys())
    doc_embeddings = np.stack([np.mean(doc_vecs[d], axis=0) for d in doc_ids], axis=0)
    doc_embeddings = doc_embeddings / (np.linalg.norm(doc_embeddings, axis=1, keepdims=True) + 1e-12)

    doc_neighbors = build_knn_graph(doc_embeddings, doc_ids, k=5)
    graph_retriever = GraphRetriever(base_retriever, doc_neighbors, chunks)

    pipeline = RAGPipeline(retriever=graph_retriever, model=llama_model, top_k=top_k)
    return pipeline, corpus, queries, qrels

from generation.agentic_pipeline import AgenticPipeline

def build_scifact_agentic_rag_pipeline(
    model_name: str = "pritamdeka/S-PubMedBert-MS-MARCO",
    chunker: str = "sent",
    llama_model: str = "llama3.2:3b",
    top_k: int = 5,
) -> Tuple[AgenticPipeline, dict, dict, dict]:
    """
    SciFact RAG pipeline with an agentic query rewriting loop.
    """
    base_retriever, corpus, queries, qrels = build_scifact_dense_retriever(
        model_name=model_name,
        chunker=chunker,
    )
    pipeline = AgenticPipeline(
        retriever=base_retriever,
        model=llama_model,
        top_k=top_k,
        rewrite_model=llama_model,
        low_conf_threshold=0.35,
    )
    return pipeline, corpus, queries, qrels