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
        raise NotImplementedError("No parser assigned ...")

    def __call__(
        self, **kwargs
    ) -> Union[List[AnnotationTuple], List[TokenTuple]]:
        return self.parse(**kwargs)


class AnnotationParser(Parser):
    """Base parser to generate list of AnnotationTuples."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.annotations: List[AnnotationTuple] = None

    def parse(self, **kwargs) -> List[AnnotationTuple]:
        raise NotImplementedError("No parser assigned ...")


class TokenParser(Parser):
    """Base parser to generate list of TokenTuples."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tokens: List[TokenTuple] = None

    def parse(self, **kwargs) -> List[TokenTuple]:
        raise NotImplementedError("No parser assigned ...")
