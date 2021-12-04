"""Contain utilities to handle vocab."""
from abc import ABC
from collections import Counter, defaultdict
from typing import Dict, List, Union

from ner_ehr.data import Constants
from ner_ehr.data.variables import (AnnotationTuple, LongAnnotation,
                                    LongAnnotationTuple)


class TokenEntityVocab(ABC):
    """Creates a vocabulary from list of annotated tuples.

    Note: indexes for unknown and pad token are reserved.
    """

    PAD_TOKEN = Constants.PAD_TOKEN.value
    PAD_TOKEN_IDX = Constants.PAD_TOKEN_IDX.value
    UNK_TOKEN = Constants.UNK_TOKEN.value
    UNK_TOKEN_IDX = Constants.UNK_TOKEN_IDX.value
    UNTAG_ENTITY_LABEL = Constants.UNTAG_ENTITY_LABEL.value
    UNTAG_ENTITY_INT_LABEL = Constants.UNTAG_ENTITY_INT_LABEL.value

    def __init__(
        self,
    ):

        # stores all unique tokens
        self.uniq_tokens = set(
            [TokenEntityVocab.PAD_TOKEN, TokenEntityVocab.UNK_TOKEN]
        )
        self.num_uniq_tokens: int = len(self.uniq_tokens)

        # stores token to idx mapping
        self._token_to_idx: Dict[str, int] = defaultdict(
            lambda: TokenEntityVocab.UNK_TOKEN_IDX
        )
        self._token_to_idx.update(
            {
                TokenEntityVocab.PAD_TOKEN: TokenEntityVocab.PAD_TOKEN_IDX,
                TokenEntityVocab.UNK_TOKEN: TokenEntityVocab.UNK_TOKEN_IDX,
            }
        )

        # stores idx to token mapping
        self._idx_to_token: Dict[int, str] = defaultdict(
            lambda: TokenEntityVocab.UNK_TOKEN
        )
        self._idx_to_token.update(
            {
                TokenEntityVocab.PAD_TOKEN_IDX: TokenEntityVocab.PAD_TOKEN,
                TokenEntityVocab.UNK_TOKEN_IDX: TokenEntityVocab.UNK_TOKEN,
            }
        )

        # stores unique entities
        self.uniq_entities = set([TokenEntityVocab.UNTAG_ENTITY_LABEL])
        self.num_uniq_entities = len(self.uniq_entities)

        # stores entity count for all tokens
        self.token_entity_freq = defaultdict(Counter)

        # stores entity to label mapping
        self._entity_to_label: Dict[str:int] = defaultdict(
            lambda: TokenEntityVocab.UNTAG_ENTITY_INT_LABEL
        )
        self._entity_to_label.update(
            {
                TokenEntityVocab.UNTAG_ENTITY_LABEL: TokenEntityVocab.UNTAG_ENTITY_INT_LABEL
            }
        )

        # stores label to entity mapping
        self._label_to_entity: Dict[str:int] = defaultdict(
            lambda: TokenEntityVocab.UNTAG_ENTITY_LABEL
        )
        self._label_to_entity.update(
            {
                TokenEntityVocab.UNTAG_ENTITY_INT_LABEL: TokenEntityVocab.UNTAG_ENTITY_LABEL
            }
        )

        # stores token to document mapping
        self.__token_doc_freq = defaultdict(set)
        self.token_doc_freq = defaultdict(int)

    @property
    def token_doc_freq(self) -> Dict[str, int]:
        """Getter for `token_doc_freq`.
        Maps token to number of documents in which that token is present.
        """
        return {
            token: len(doc_ids)
            for token, doc_ids in self.__token_doc_freq.items()
        }

    @token_doc_freq.setter
    def token_doc_freq(self, arg) -> Dict[str, int]:
        pass

    def _add_token(self, token: str) -> None:
        """Helper function to add token and assign it's index."""
        self.uniq_tokens.add(token)
        if len(self.uniq_tokens) > self.num_uniq_tokens:  # new token added
            idx = self.num_uniq_tokens
            # assigning unique idx to each token
            self._token_to_idx.update({token: idx})
            self._idx_to_token.update({idx: token})
            self.num_uniq_tokens += 1

    def _add_entity(self, entity: str) -> None:
        """Helper function to add entity and assign it's label"""
        self.uniq_entities.add(entity)

        if len(self.uniq_entities) > self.num_uniq_entities:
            # new entity added
            label = self.num_uniq_entities
            self._entity_to_label.update({entity: label})
            self._label_to_entity.update({label: entity})
            self.num_uniq_entities += 1

    def fit(self, annotatedtuples: List[AnnotationTuple]) -> None:
        """Generate vocab, token to idx mapping,
            token to entity (with entity count) mapping,
            token to document mapping from list of annotated tokens.

        Args:
            annotatedtuples: list of AnnotatedToken tuples
                [
                    Annotation(
                        doc_id='100035',
                        token='Admission',
                        start_idx=0,
                        end_idx=9,
                        entity='O'),
                    Annotation(
                        doc_id='100035',
                        token='Date',
                        start_idx=10,
                        end_idx=14,
                        entity='O'),
                ]
        """
        for token in annotatedtuples:
            self._add_token(token=token.token)
            self._add_entity(entity=token.entity)
            self.token_entity_freq[token.token].update({token.entity: 1})
            self.__token_doc_freq[token.token].add(token.doc_id)

    def _validate_list(self, l: List[Union[int, str]], dtype: type):
        """Helper function to validate given type of input and it's value."""
        if not isinstance(l, list):
            raise TypeError(f"Expect input to be a list, not {type(l)}")

        if not all(isinstance(item, dtype) for item in l):
            raise TypeError(
                f"Expected dtype of all items in list to be {dtype}"
            )

    def token_to_idx(self, tokens: List[str]) -> List[int]:
        """Convert list of string tokens into list of integer (index) tokens.

        Note: indexes for unkown tokens(token not in vocab)
            are replaced by `unkown` token index.

        Args:
            tokens: list of string tokens
                ["Admission", "data"]

        Returns:
            idxs: list of integer (index) token
                [2, 3]
        """
        self._validate_list(tokens, str)
        return [self._token_to_idx[token] for token in tokens]

    def idx_to_token(self, idxs: List[int]) -> List[str]:
        """Convert list of integer (index) tokens into list of string tokens.

        Note: indexes for unknown tokens(token not in vocab)
            are replaced by `unknown` token index.

        Args:
            idxs: list of integer (index) tokens
                [2, 3]

        Returns:
            tokens: list of string tokens
                ["Admission", "data"]
        """
        self._validate_list(idxs, int)
        return [self._idx_to_token[idx] for idx in idxs]

    def entity_to_label(self, entities: List[str]) -> List[int]:
        """Convert list of string entities into list of integer (label) entities.

        Args:
            entities: list of string entities
                ["B-Frequency", "I-Frequency"]

        Returns:
            labels: list of integer (label) entities
                [2, 3]
        """
        self._validate_list(entities, str)
        return [self._entity_to_label[entity] for entity in entities]

    def label_to_entity(self, labels: List[int]) -> List[str]:
        """Convert list of integer (label) entities into list of string entities.

        Args:
            labels: list of integer (label) entities
                [2, 3]

        Returns:
            entities: list of string entities
                ["B-Frequency", "I-Frequency"]
        """
        self._validate_list(labels, int)
        return [self._label_to_entity[label] for label in labels]

    def annotation_to_longannotation(
        self, annotatedtuple: AnnotationTuple
    ) -> LongAnnotationTuple:
        """Convert AnnotationTuple to LongAnnotationTuple.

        Args:
            annotatedtuple: An annotation tuple
                Ex: Annotation(
                        doc_id='100035',
                        token='Admission',
                        start_idx=0,
                        end_idx=9,
                        entity='O')

        Returns:
            long_annotatedtuple: A long annotated tuple
                Ex: Annotation(
                        doc_id='100035',
                        token='Admission',
                        start_idx=0,
                        end_idx=9,
                        entity='O',
                        token_idx: 2,
                        entity_label=0)
        """
        return LongAnnotation(
            doc_id=annotatedtuple.doc_id,
            token=annotatedtuple.token,
            start_idx=annotatedtuple.start_idx,
            end_idx=annotatedtuple.end_idx,
            entity=annotatedtuple.entity,
            token_idx=self._token_to_idx[annotatedtuple.token],
            entity_label=self._entity_to_label[annotatedtuple.entity],
        )()