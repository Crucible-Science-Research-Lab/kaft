"""
kaft.ingest.router — Corpus routers for real-world domains.

Each router fetches documents and returns records in the standard kaft format:
    list[RouterRecord] — ready to pass directly into build_manifold().

Forward-compatible design:
    BaseRouter defines the interface.
    ArxivRouter, PubMedRouter, PatentRouter all slot in identically.
    Downstream code (build_manifold, experiment scripts) never imports a
    specific router — only the interface.

RouterRecord is the explicit contract between ingest and the manifold layer.
All keys that build_manifold() and experiment scripts depend on live here.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TypedDict
import logging
import arxiv

log = logging.getLogger(__name__)


# ── Shared record format ──────────────────────────────────────────────────────

class RouterRecord(TypedDict):
    """
    Standard kaft ingest record.

    Every router returns list[RouterRecord] — same contract regardless of source.

    text     : what build_manifold() embeds  (title + abstract for arXiv)
    title    : human-readable label for topology map
    abstract : full source text (may equal text for some routers)
    source   : router identifier ("arxiv", "pubmed", "patent", ...)
    source_id: unique id within source (arxiv_id, PMID, patent number)
    authors  : list of author/inventor names
    url      : canonical web URL for the document
    """
    text:      str
    title:     str
    abstract:  str
    source:    str
    source_id: str
    authors:   list[str]
    url:       str


# ── Abstract base ─────────────────────────────────────────────────────────────

class BaseRouter(ABC):
    """
    Interface contract for all kaft ingest routers.

    Subclasses implement fetch() — everything else (batch sweeps,
    logging, downstream compatibility) is handled here.
    """

    @abstractmethod
    def fetch(
        self,
        query:       str,
        max_results: int | None = None,
        silent:      bool       = False,
    ) -> list[RouterRecord]:
        """
        Fetch documents matching query.

        Parameters
        ----------
        query       : domain-specific search string
        max_results : overrides instance default if provided
        silent      : suppress logging (use True for batch/sweep runs)

        Returns
        -------
        list[RouterRecord] — ready for build_manifold()
        """

    def fetch_batch(
        self,
        query:  str,
        sizes:  list[int],
        silent: bool = True,
    ) -> dict[int, list[RouterRecord]]:
        """
        Fetch at multiple corpus sizes for N-scaling experiments.

        Fetches max(sizes) once, then slices — minimises API calls.

        Parameters
        ----------
        query  : search string
        sizes  : e.g. [30, 50, 100, 150, 200]
        silent : suppress per-fetch logging (default True for sweeps)

        Returns
        -------
        dict mapping N -> list[RouterRecord] of exactly N records
        """
        n_max   = max(sizes)
        records = self.fetch(query, max_results=n_max, silent=silent)

        if not silent:
            log.info(f"[{self.__class__.__name__}] batch '{query}' — {len(records)} fetched, "
                     f"slicing to {sizes}")

        return {n: records[:n] for n in sizes if n <= len(records)}


# ── ArXiv router ──────────────────────────────────────────────────────────────

class ArxivRouter(BaseRouter):
    """
    Fetch papers from arXiv → list[RouterRecord].

    Usage
    -----
    router  = ArxivRouter()
    records = router.fetch("attention mechanism transformer", max_results=100)
    state   = build_manifold(records)

    N-scaling sweep (for exp_arxiv_softmax_vs_fr):
    batches = router.fetch_batch("attention mechanism", sizes=[30,50,100,150,200])
    # batches[100] → exactly 100 RouterRecords
    """

    def __init__(
        self,
        max_results:   int   = 50,
        delay_seconds: float = 5.0,
        page_size:     int   = 100,
    ):
        self.max_results = max_results
        self._client     = arxiv.Client(
            num_retries   = 5,
            delay_seconds = delay_seconds,
            page_size     = page_size,
        )


    def fetch(
        self,
        query:       str,
        max_results: int | None = None,
        silent:      bool       = False,
    ) -> list[RouterRecord]:
        """
        Fetch arXiv papers matching query.

        text field = title + ". " + abstract — this is what build_manifold embeds.
        source_id  = short arXiv ID (e.g. "2301.07041").
        """
        n      = max_results or self.max_results
        search = arxiv.Search(
            query      = query,
            max_results = n,
            sort_by    = arxiv.SortCriterion.Relevance,
        )

        records: list[RouterRecord] = []
        for paper in self._client.results(search):
            records.append(RouterRecord(
                text      = f"{paper.title}. {paper.summary}",
                title     = paper.title,
                abstract  = paper.summary,
                source    = "arxiv",
                source_id = paper.get_short_id(),
                authors   = [a.name for a in paper.authors],
                url       = paper.entry_id,
            ))

        if not silent:
            log.info(f"[ArxivRouter] '{query}' -> {len(records)} papers fetched")

        return records
