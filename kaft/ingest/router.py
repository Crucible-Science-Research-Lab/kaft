"""
kaft.ingest — corpus routers for real-world domains.

Each router fetches documents and returns records in the
standard kaft format: list of dicts with "text" key.
Ready to pass directly into build_manifold().
"""
from __future__ import annotations
import arxiv


class ArxivRouter:
    """
    Fetch papers from arXiv and return kaft-compatible records.

    Each record contains full metadata so the manifold knows
    not just the embedding but the source paper.

    Usage
    -----
    router = ArxivRouter()
    records = router.fetch("CRISPR gene editing", max_results=15)
    state = build_manifold(records)
    """

    def __init__(self, max_results: int = 20):
        self.max_results = max_results
        self.client = arxiv.Client(
            num_retries=5,
            delay_seconds=5.0
        )

    def fetch(self, query: str, max_results: int | None = None) -> list[dict]:
        """
        Fetch arXiv papers matching query.

        Parameters
        ----------
        query       : arXiv search string
        max_results : overrides instance default if provided

        Returns
        -------
        list of dicts, each with keys:
            text     — title + abstract (what build_manifold embeds)
            title    — paper title
            abstract — full abstract
            arxiv_id — e.g. "2301.07041"
            authors  — list of author names
            url      — arXiv abs URL
        """
        n = max_results or self.max_results

        search = arxiv.Search(
            query=query,
            max_results=n,
            sort_by=arxiv.SortCriterion.Relevance
        )

        records = []
        for paper in self.client.results(search):
            records.append({
                "text"    : f"{paper.title}. {paper.summary}",
                "title"   : paper.title,
                "abstract": paper.summary,
                "arxiv_id": paper.get_short_id(),
                "authors" : [a.name for a in paper.authors],
                "url"     : paper.entry_id
            })

        print(f"[ArxivRouter] '{query}' → {len(records)} papers fetched")
        return records
