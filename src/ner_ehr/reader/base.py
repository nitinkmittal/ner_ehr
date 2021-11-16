from abc import ABC
from pathlib import Path
from typing import Union


class BaseReader(ABC):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def read(str, fp: Union[str, Path]):
        pass
