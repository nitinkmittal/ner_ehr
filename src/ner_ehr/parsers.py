"""This module contains parser definitions."""
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
        """
        Base callable parser.

        Args:
            **kwargs: keyword-arguments

        Returns:
            A list of tokentuples or annotatedtuples
        """
        raise NotImplementedError("Parser not implemented ...")

    def __call__(
        self, **kwargs
    ) -> Union[List[AnnotationTuple], List[TokenTuple]]:
        return self.parse(**kwargs)


class AnnotationParser(Parser):
    """Base parser to generate list of annotated tuples."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.annotations: List[AnnotationTuple] = None

    def parse(self, **kwargs) -> List[AnnotationTuple]:
        raise NotImplementedError("Parser not implemented ...")


class TokenParser(Parser):
    """Base parser to generate list of token (unannotated) tuples."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tokens: List[TokenTuple] = None

    def parse(self, **kwargs) -> List[TokenTuple]:
        raise NotImplementedError("Parser not implemented ...")
