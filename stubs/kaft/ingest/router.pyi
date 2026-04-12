import abc
from _typeshed import Incomplete
from abc import ABC, abstractmethod
from typing import TypedDict

log: Incomplete

class RouterRecord(TypedDict):
    text: str
    title: str
    abstract: str
    source: str
    source_id: str
    authors: list[str]
    url: str

class BaseRouter(ABC, metaclass=abc.ABCMeta):
    @abstractmethod
    def fetch(self, query: str, max_results: int | None = None, silent: bool = False) -> list[RouterRecord]: ...
    def fetch_batch(self, query: str, sizes: list[int], silent: bool = True) -> dict[int, list[RouterRecord]]: ...

class ArxivRouter(BaseRouter):
    max_results: Incomplete
    def __init__(self, max_results: int = 50, delay_seconds: float = 5.0, page_size: int = 100) -> None: ...
    def fetch(self, query: str, max_results: int | None = None, silent: bool = False) -> list[RouterRecord]: ...
