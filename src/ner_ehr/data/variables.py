"""Contain definitions for entities: tokens and annotations."""
from abc import ABC
from collections import namedtuple
from typing import NamedTuple


class Variable(ABC):
    """Base variable."""

    def __init__(
        self, doc_id: str, token: str, start_idx: int, end_idx: int, **kwargs
    ):
        """
        Args:
            doc_id: unique identifier for document/sample from which
                variable is generated

            token: value of the variable

            start_idx: index of first character of token picked from text

            end_idx: index of last character of token picked from text

            **kwargs: other keyword arguments
        """

        self.doc_id = doc_id
        self.token = token
        self.start_idx = start_idx
        self.end_idx = end_idx

        self.__dict__.update(kwargs)
        self.tuple: NamedTuple = None

    @property
    def doc_id(self) -> str:
        return self._doc_id

    @doc_id.setter
    def doc_id(self, idx):
        self._doc_id = str(idx)

    @property
    def token(self) -> str:
        return self._token

    @token.setter
    def token(self, idx):
        self._token = str(idx)

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


TokenTuple = namedtuple(
    "Token", field_names=["doc_id", "token", "start_idx", "end_idx"]
)


class Token(Variable):
    """Definition of a token."""

    def __init__(self, doc_id: str, token: str, start_idx: int, end_idx: int):
        super().__init__(
            doc_id=doc_id,
            token=token,
            start_idx=start_idx,
            end_idx=end_idx,
        )

        self.tuple: TokenTuple = TokenTuple(
            doc_id=self.doc_id,
            token=self.token,
            start_idx=self.start_idx,
            end_idx=self.end_idx,
        )


AnnotationTuple = namedtuple(
    "Annotation",
    field_names=["doc_id", "token", "start_idx", "end_idx", "entity"],
)


class Annotation(Variable):
    """Definition of an annotation."""

    def __init__(
        self,
        doc_id: str,
        token: str,
        start_idx: int,
        end_idx: int,
        entity: str,
    ):
        super().__init__(
            doc_id=doc_id,
            token=token,
            start_idx=start_idx,
            end_idx=end_idx,
        )
        self.entity = entity

        self.tuple: AnnotationTuple = AnnotationTuple(
            doc_id=self.doc_id,
            token=self.token,
            start_idx=self.start_idx,
            end_idx=self.end_idx,
            entity=self.entity,
        )

    @property
    def entity(self) -> str:
        return self._entity

    @entity.setter
    def entity(self, entity):
        self._entity = str(entity)


LongAnnotationTuple = namedtuple(
    "LongAnnotation",
    field_names=[
        "doc_id",
        "token",
        "start_idx",
        "end_idx",
        "entity",
        "token_idx",
    ],
)


class LongAnnotation(Annotation):
    """Definition of an annotation."""

    def __init__(
        self,
        doc_id: str,
        token: str,
        start_idx: int,
        end_idx: int,
        entity: str,
        token_idx: int,
    ):
        super().__init__(
            doc_id=doc_id,
            token=token,
            start_idx=start_idx,
            end_idx=end_idx,
            entity=entity,
        )

        self.token_idx = token_idx

        self.tuple: LongAnnotationTuple = LongAnnotationTuple(
            doc_id=self.doc_id,
            token=self.token,
            start_idx=self.start_idx,
            end_idx=self.end_idx,
            entity=self.entity,
            token_idx=self.token_idx,
        )

    @property
    def token_idx(self) -> int:
        return self._token_idx

    @token_idx.setter
    def token_idx(self, idx: int):
        self._token_idx = int(idx)
