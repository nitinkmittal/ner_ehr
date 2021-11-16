from abc import ABC
from typing import Callable, List, Tuple


class BaseTokenizer(ABC):
    def __init__(self, tokenizer: Callable[[str], List[str]]):
        self.tokenizer = tokenizer

    def _tokenize(self, text: str) -> List[Tuple[str, Tuple[int, int]]]:
        pass

    def __call__(self, text: str):
        return self.tokenize(text)
