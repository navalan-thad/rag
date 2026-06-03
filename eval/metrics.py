def recall_at_k(retrieved_doc_ids: list, relevant_doc_ids: set, k: int) -> float:
    return int(any(d in relevant_doc_ids for d in retrieved_doc_ids[:k]))

def reciprocal_rank(retrieved_doc_ids: list, relevant_doc_ids: set) -> float:
    for i, doc_id in enumerate(retrieved_doc_ids):
        if doc_id in relevant_doc_ids:
            return 1.0 / (i + 1)
    return 0.0

def evaluate(retriever, queries, qrels, k_values=[1, 5, 10]):
    recall = {k: [] for k in k_values}
    mrr_scores = []

    for query_id, query_text in queries.items():
        relevant = set(qrels.get(query_id, {}).keys())
        if not relevant:
            continue
        hits = retriever.retrieve(query_text, top_k=max(k_values))
        retrieved_ids = [h["doc_id"] for h in hits]

        for k in k_values:
            recall[k].append(recall_at_k(retrieved_ids, relevant, k))
        mrr_scores.append(reciprocal_rank(retrieved_ids, relevant))

    results = {f"Recall@{k}": round(sum(v)/len(v), 4) for k, v in recall.items()}
    results["MRR"] = round(sum(mrr_scores)/len(mrr_scores), 4)
    return results

def log_failures(retriever, queries, qrels, n=20):
    failures = []
    for query_id, query_text in list(queries.items())[:200]:
        relevant = set(qrels.get(query_id, {}).keys())
        hits = retriever.retrieve(query_text, top_k=5)
        retrieved_ids = [h["doc_id"] for h in hits]
        if not any(d in relevant for d in retrieved_ids):
            failures.append({
                "query": query_text,
                "relevant_docs": list(relevant),
                "retrieved": [(h["doc_id"], round(h["score"],3)) for h in hits]
            })
    return failures[:n]

def categorize_failures(retriever, queries, qrels, corpus, n=50):
    results = {"lexical_gap": [], "semantic_drift": [], "low_confidence": [], "other": []}

    for query_id, query_text in list(queries.items()):
        relevant = set(qrels.get(query_id, {}).keys())
        if not relevant:
            continue
        hits = retriever.retrieve(query_text, top_k=10)
        retrieved_ids = [h["doc_id"] for h in hits]

        if any(d in relevant for d in retrieved_ids):
            continue  # not a failure

        top_score = hits[0]["score"] if hits else 0
        query_words = set(query_text.lower().split())

        # low confidence: top score below threshold
        if top_score < 0.35:
            category = "low_confidence"
        # check if any retrieved doc shares rare query terms (lexical gap signal)
        elif not any(
            query_words & set(corpus[h["doc_id"]]["text"].lower().split())
            for h in hits[:3]
        ):
            category = "lexical_gap"
        # high confidence wrong answer = semantic drift
        elif top_score > 0.60:
            category = "semantic_drift"
        else:
            category = "other"

        results[category].append({
            "query": query_text,
            "relevant": list(relevant),
            "top_hit": hits[0] if hits else None,
            "top_score": top_score
        })

    # trim to n total
    for k in results:
        results[k] = results[k][:n // 4]

    return results