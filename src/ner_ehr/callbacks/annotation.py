from typing import List, Callable
from .base import BaseCallback
from .base import ABC


class Parser(BaseCallback):
    """Parser to process annotations."""

    def __init__(self, parser: Callable[[str], List[str]]):
        """
        Args:
            sep: separator to split string into list of strings.
        """
        super().__init__(parser=parser)

    def __call__(self, text: str) -> List[str]:
        """Callable function"""
        return self.parser(text)


class Validator(BaseCallback):
    """Validate an annotation for given validation condition."""

    def __init__(self, validator: Callable[[str], bool]):
        super().__init__(validator=validator)

    def __call__(self, annotation: List[str]) -> bool:
        return self.validator(annotation)
