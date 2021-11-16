from abc import ABC
from typing import Any


class BaseCallback(ABC):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __call__(self, **kwargs) -> Any:
        pass
