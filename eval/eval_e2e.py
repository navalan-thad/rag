import json
from pathlib import Path
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_correctness
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig
from langchain_ollama import ChatOllama, OllamaEmbeddings
from generation.rag_pipeline import RAGPipeline


def build_ragas_dataset(results, ground_truths):
    return Dataset.from_list([
        {
            "question":     r.question,
            "answer":       r.answer,
            "contexts":     r.contexts,
            "ground_truth": gt,
        }
        for r, gt in zip(results, ground_truths)
    ])


def run_e2e_evaluation(pipeline: RAGPipeline, queries, ground_truths, output_path: str):
    results = pipeline.run_batch(queries)

    judge_llm = LangchainLLMWrapper(ChatOllama(model="qwen2.5:14b", temperature=0))
    judge_emb = LangchainEmbeddingsWrapper(OllamaEmbeddings(model="nomic-embed-text"))

    dataset = build_ragas_dataset(results, ground_truths)

    scores = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_correctness],
        llm=judge_llm,
        embeddings=judge_emb,
        run_config=RunConfig(max_workers=1, timeout=120),
    )

    scores_df = scores.to_pandas()
    faithfulness_score       = float(scores_df["faithfulness"].mean())
    # answer_correctness_score = float(scores_df["answer_correctness"].mean())

    output = {
        "faithfulness":       faithfulness_score,
        # "answer_correctness": answer_correctness_score,
        "per_question": [
            {
                "question":     r.question,
                "answer":       r.answer,
                "source_ids":   r.source_ids,
                "ground_truth": gt,
            }
            for r, gt in zip(results, ground_truths)
        ]
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(output, indent=2))
    print(f"Faithfulness:       {faithfulness_score:.4f}")
    # print(f"Answer Correctness: {answer_correctness_score:.4f}")
    return output