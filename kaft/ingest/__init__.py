"""
kaft.ingest — corpus routers for real-world domains.

Exports the shared contract (RouterRecord, BaseRouter) and all
concrete router implementations so downstream code imports cleanly:

    from kaft.ingest import ArxivRouter, RouterRecord
"""
from kaft.ingest.router import RouterRecord, BaseRouter, ArxivRouter

__all__ = ["RouterRecord", "BaseRouter", "ArxivRouter"]