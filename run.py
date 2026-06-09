import config

from pipelines import build_scifact_dense_retriever
from eval.metrics import evaluate, categorize_failures


def main():
    retriever, corpus, queries, qrels = build_scifact_dense_retriever(
        model_name="all-MiniLM-L6-v2",
        chunker="whole",
    )

    results = evaluate(retriever, queries, qrels, k_values=[1, 5, 10])
    print(results)

    failure_report = categorize_failures(retriever, queries, qrels, corpus, n=60)
    for category, cases in failure_report.items():
        print(f"\n{category.upper()} ({len(cases)} cases)")
        for f in cases[:3]:
            print(f"  Query: {f['query']}")
            print(f"  Score: {f['top_score']:.3f}")

if __name__ == "__main__":
    main()