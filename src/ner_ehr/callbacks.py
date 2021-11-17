"""Contain callable function definations."""
from abc import ABC
from typing import Any, Callable, List, Union
from pathlib import Path
from ner_ehr.data.entities import AnnotationTuple


class Callback(ABC):
    """Base callback."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __call__(self, **kwargs) -> Any:
        pass


class AnnotationParser(Callback):
    """Parser to process an annotation."""

    def __init__(
        self, parser: Callable[[Union[str, Path], Any], List[AnnotationTuple]]
    ):
        """
        Args:
            parser: parser (callable): to build list of annotations from string.
            >>> parser("T1	Reason 10179 10197	recurrent seizures") ->
                [
                    Annotation(token='recurrent', start_idx=10179, end_idx=10188, tag='B-Reason'),
                    Annotation(token='seizures', start_idx=10189, end_idx=10197, tag='I-Reason')
                ]
        """
        super().__init__(parser=parser)

    def __call__(self, **kwargs) -> List[AnnotationTuple]:
        """Callable function"""
        return self.parser(**kwargs)
