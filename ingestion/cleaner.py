def clean_doc(doc_id, doc):
    text = f"{doc['title']}. {doc['text']}".strip()
    return {"id": doc_id, "text": text}

def build_clean_corpus(corpus):
    return [clean_doc(k, v) for k, v in corpus.items()]