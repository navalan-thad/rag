from ingestion.loader import load_scifact
from ingestion.cleaner import build_clean_corpus
from experiments.run_full_rag import pipeline
import random

corpus, queries, qrels = load_scifact()
docs = build_clean_corpus(corpus)

sample_qids = random.sample(list(queries.keys()), 20)

for qid in sample_qids:
    print("=" * 80)
    print("QID:", qid)
    print("Question:", queries[qid])

    relevant = qrels.get(qid, {})
    if relevant:
        best_doc_id = max(relevant, key=relevant.get)
        print("\nGold doc:", corpus[best_doc_id].get("title", ""))
        print(corpus[best_doc_id]["text"][:500], "...\n")

    result = pipeline.run(queries[qid])
    print("Answer:", result.answer)
    print("\nContexts:")
    for i, c in enumerate(result.contexts):
        print(f"[{i+1}]", c[:400].replace("\n", " "), "...\n")