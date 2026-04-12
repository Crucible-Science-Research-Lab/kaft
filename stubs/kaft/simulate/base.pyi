import abc
from abc import ABC, abstractmethod

class AbstractManifoldDynamics(ABC, metaclass=abc.ABCMeta):
    @abstractmethod
    def run(self, manifold: dict, dt: float = 1.0) -> dict: ...
    @abstractmethod
    def geometry_type(self) -> str: ...
