from typing import List, Dict

def chunk_whole(docs) -> List[Dict]:
    """No chunking (each doc is one chunk)"""
    return [{"chunk_id": d["id"], "doc_id": d["id"], "text": d["text"]} for d in docs]

def chunk_fixed(docs, size=200, overlap=20) -> List[Dict]:
    """Fixed token-window chunking."""
    chunks = []
    for d in docs:
        words = d["text"].split()
        start = 0
        i = 0
        while start < len(words):
            chunk_text = " ".join(words[start:start+size])
            chunks.append({"chunk_id": f"{d['id']}_{i}", "doc_id": d["id"], "text": chunk_text})
            start += size - overlap
            i += 1
    return chunks

def chunk_sentences(docs, max_sentences=5) -> List[Dict]:
    """Sentence-boundary chunking."""
    import re
    chunks = []
    for d in docs:
        sentences = re.split(r'(?<=[.!?])\s+', d["text"])
        for i in range(0, len(sentences), max_sentences):
            chunk_text = " ".join(sentences[i:i+max_sentences])
            chunks.append({"chunk_id": f"{d['id']}_{i}", "doc_id": d["id"], "text": chunk_text})
    return chunks