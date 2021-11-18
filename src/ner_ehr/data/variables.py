"""Contain definitions for entities: tokens and annotations."""
from abc import ABC
from typing import NamedTuple, List, Union
from collections import namedtuple

import pandas as pd


class Variable(ABC):
    """Base entity."""

    def __init__(
        self, name: str, token: str, start_idx: int, end_idx: int, **kwargs
    ):
        """
        Args:
            name: name of the entity

            token: value of the entity

            start_idx: index of first character of token picked from text

            end_idx: index of last character of token picked from text

            **kwargs: other keyword arguments
        """
        self.name = name
        self.token = token
        self.start_idx = start_idx
        self.end_idx = end_idx

        self.__dict__.update(kwargs)
        self.tuple: NamedTuple = None

    @property
    def start_idx(self) -> int:
        return self._start_idx

    @start_idx.setter
    def start_idx(self, idx):
        self._start_idx = int(idx)

    @property
    def end_idx(self):
        return self._end_idx

    @end_idx.setter
    def end_idx(self, idx) -> int:
        self._end_idx = int(idx)

    def __call__(
        self,
    ) -> NamedTuple:
        return self.tuple


TokenTuple = namedtuple("Token", field_names=["token", "start_idx", "end_idx"])


class Token(Variable):
    """Definition of a token."""

    def __init__(self, token: str, start_idx: int, end_idx: int):
        super().__init__(
            name="Token", token=token, start_idx=start_idx, end_idx=end_idx
        )

        self.tuple: TokenTuple = TokenTuple(
            token=self.token, start_idx=self.start_idx, end_idx=self.end_idx
        )


AnnotationTuple = namedtuple(
    "Annotation", field_names=["token", "start_idx", "end_idx", "tag"]
)


class Annotation(Variable):
    """Definition of an annotation."""

    def __init__(self, token: str, start_idx: int, end_idx: int, tag: str):
        super().__init__(
            name="Annotation",
            token=token,
            start_idx=start_idx,
            end_idx=end_idx,
            tag=tag,
        )

        self.tuple: AnnotationTuple = AnnotationTuple(
            token=self.token,
            start_idx=self.start_idx,
            end_idx=self.end_idx,
            tag=self.tag,
        )
