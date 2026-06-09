from typing import List, Dict
import re

def chunk_whole(docs: List[Dict]) -> List[Dict]:
    """
    No chunking; each doc is one chunk.
    Assumes each doc is {"id": doc_id, "text": str}.
    """
    chunks = []
    for d in docs:
        doc_id = d["id"]
        chunks.append({
            "chunk_id": doc_id,
            "doc_id": doc_id,
            "text": d["text"],
        })
    return chunks


def chunk_fixed(docs: List[Dict], size: int = 200, overlap: int = 20) -> List[Dict]:
    """
    Fixed token-window chunking by whitespace tokens.
    """
    chunks = []
    for d in docs:
        doc_id = d["id"]
        words = d["text"].split()
        start = 0
        i = 0
        while start < len(words):
            chunk_words = words[start:start + size]
            chunk_text = " ".join(chunk_words)
            chunks.append({
                "chunk_id": f"{doc_id}_{i}",
                "doc_id": doc_id,
                "text": chunk_text,
            })
            start += size - overlap
            i += 1
    return chunks


def chunk_sentences(docs: List[Dict], max_sentences: int = 5) -> List[Dict]:
    """
    Sentence-boundary chunking.
    """
    chunks = []
    for d in docs:
        doc_id = d["id"]
        sentences = re.split(r"(?<=[.!?])\s+", d["text"])
        for i in range(0, len(sentences), max_sentences):
            chunk_text = " ".join(sentences[i:i + max_sentences])
            if not chunk_text.strip():
                continue
            chunks.append({
                "chunk_id": f"{doc_id}_{i}",
                "doc_id": doc_id,
                "text": chunk_text,
            })
    return chunks