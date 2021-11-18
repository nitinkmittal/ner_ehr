"""Contain callable function definations."""
from abc import ABC
from typing import List, Union
from ner_ehr.data.variables import AnnotationTuple, TokenTuple


class Parser(ABC):
    """Base parser."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def parse(
        self, **kwargs
    ) -> Union[List[AnnotationTuple], List[TokenTuple]]:
        pass

    def __call__(
        self, **kwargs
    ) -> Union[List[AnnotationTuple], List[TokenTuple]]:
        return self.parse(**kwargs)


class AnnotationParser(Parser):
    """Base parser to generate list of AnnotationTuple."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.annotations: List[AnnotationTuple] = []

    def parse(self, **kwargs) -> List[AnnotationTuple]:
        pass


class TokenParser(Parser):
    """Base parser to generate list of TokenTuple."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tokens: List[TokenTuple] = []

    def parse(self, **kwargs) -> List[TokenTuple]:
        pass
