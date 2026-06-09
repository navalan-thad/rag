from typing import Dict, Tuple, Any
import arxiv

Corpus = Dict[str, Dict[str, Any]]
Queries = Dict[str, str]


def load_arxiv_corpus(
    topic: str,
    max_results: int = 50,
) -> Tuple[Corpus, Queries]:
    search = arxiv.Search(
        query=topic,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )

    # Use the new Client API
    client = arxiv.Client(
        page_size=50,        # how many per API call
        delay_seconds=3,     # be nice to arXiv
        num_retries=2,
    )

    corpus: Corpus = {}
    queries: Queries = {}

    for i, result in enumerate(client.results(search)):
        if i >= max_results:
            break

        doc_id = str(i)
        title = result.title.strip()
        text = f"{result.title.strip()}\n\n{result.summary.strip()}"

        corpus[doc_id] = {
            "title": title,
            "text": text,
            "url": result.entry_id,
            "published": result.published.strftime("%Y-%m-%d") if result.published else "",
        }

    return corpus, queries