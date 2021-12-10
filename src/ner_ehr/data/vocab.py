"""This module definition of TokenEntityVocab."""
from abc import ABC
from collections import Counter, defaultdict
from typing import Counter as typeCounter
from typing import DefaultDict, Dict, List, Set, Union

from ner_ehr.data import Constants
from ner_ehr.data.variables import (
    AnnotationTuple,
    LongAnnotation,
    LongAnnotationTuple,
)
from ner_ehr.utils import validate_list

TO_LOWER: bool = False


class TokenEntityVocab(ABC):
    """Creates a vocabulary from list of annotated tuples.

    Note: indexes for unknown and pad token are reserved.
    """

    PAD_TOKEN = Constants.PAD_TOKEN.value
    PAD_TOKEN_IDX = Constants.PAD_TOKEN_IDX.value
    PAD_TOKEN_ENTITY_LABEL = Constants.PAD_TOKEN_ENTITY_LABEL.value
    PAD_TOKEN_ENTITY_INT_LABEL = Constants.PAD_TOKEN_ENTITY_INT_LABEL.value
    UNK_TOKEN = Constants.UNK_TOKEN.value
    UNK_TOKEN_IDX = Constants.UNK_TOKEN_IDX.value
    UNTAG_ENTITY_LABEL = Constants.UNTAG_ENTITY_LABEL.value
    UNTAG_ENTITY_INT_LABEL = Constants.UNTAG_ENTITY_INT_LABEL.value

    def __init__(
        self,
        to_lower: bool = TO_LOWER,
        ignore_tokens: List[str] = [],
    ):
        """
        Args:
            to_lower: boolean flag, default=False
                if True, tokens are lowercased before adding into vocab,
                otherwise not

            ignore_tokens: a list of string tokens used to ignore tokens
                while saving tokens, by default empty
        """
        self.to_lower = to_lower
        self.ignore_tokens = ignore_tokens

        # stores all unique tokens
        self.uniq_tokens: Set[str] = None
        # stores number of unique tokens
        self.num_uniq_tokens: int = None

        # stores token to idx mapping
        self._token_to_idx: Dict[str, int] = None
        # stores idx to token mapping
        self._idx_to_token: Dict[int, str] = None

        # stores unique entities
        self.num_uniq_entities: int = None

        # stores entity to label mapping
        self._entity_to_label: Dict[str, int] = None
        # stores label to entity mapping
        self._label_to_entity: Dict[int, str] = None

        # stores entity count for all tokens
        self.token_entity_freq: DefaultDict[str, typeCounter[str]] = None
        # stores token frequency
        self.token_freq: DefaultDict[str, typeCounter[str]] = None
        # stores token to document mapping
        self.__token_doc_freq: DefaultDict[str, Set[str]] = None
        # stores token document frequency
        self.token_doc_freq: DefaultDict[str, int] = None

    def _setup(
        self,
    ) -> None:
        """Initialize class instances/variables."""
        self.ignore_tokens = set(
            [self._to_lower(token=token) for token in self.ignore_tokens]
        )

        self.uniq_tokens: Set[str] = set(
            [TokenEntityVocab.PAD_TOKEN, TokenEntityVocab.UNK_TOKEN]
        )
        self.num_uniq_tokens: int = len(self.uniq_tokens)

        self._token_to_idx: Dict[str, int] = defaultdict(
            lambda: TokenEntityVocab.UNK_TOKEN_IDX
        )
        self._token_to_idx.update(
            {
                TokenEntityVocab.PAD_TOKEN: TokenEntityVocab.PAD_TOKEN_IDX,
                TokenEntityVocab.UNK_TOKEN: TokenEntityVocab.UNK_TOKEN_IDX,
            }
        )
        self._idx_to_token: Dict[int, str] = defaultdict(
            lambda: TokenEntityVocab.UNK_TOKEN
        )
        self._idx_to_token.update(
            {
                TokenEntityVocab.PAD_TOKEN_IDX: TokenEntityVocab.PAD_TOKEN,
                TokenEntityVocab.UNK_TOKEN_IDX: TokenEntityVocab.UNK_TOKEN,
            }
        )

        self._entity_to_label: Dict[str, int] = defaultdict(
            lambda: TokenEntityVocab.UNTAG_ENTITY_INT_LABEL
        )
        self._entity_to_label.update(
            {
                TokenEntityVocab.PAD_TOKEN_ENTITY_LABEL: TokenEntityVocab.PAD_TOKEN_ENTITY_INT_LABEL,
                TokenEntityVocab.UNTAG_ENTITY_LABEL: TokenEntityVocab.UNTAG_ENTITY_INT_LABEL,
            }
        )

        self._label_to_entity: Dict[int, str] = defaultdict(
            lambda: TokenEntityVocab.UNTAG_ENTITY_LABEL
        )
        self._label_to_entity.update(
            {
                TokenEntityVocab.PAD_TOKEN_ENTITY_INT_LABEL: TokenEntityVocab.PAD_TOKEN_ENTITY_LABEL,
                TokenEntityVocab.UNTAG_ENTITY_INT_LABEL: TokenEntityVocab.UNTAG_ENTITY_LABEL,
            }
        )

        self.token_entity_freq: DefaultDict[
            str, typeCounter[str]
        ] = defaultdict(Counter)
        self.token_freq: DefaultDict[str, Set[str]] = defaultdict()
        self.__token_doc_freq: DefaultDict[str, Set[str]] = defaultdict(set)
        self.token_doc_freq: DefaultDict[str, int] = defaultdict(int)

    @property
    def token_doc_freq(self) -> Dict[str, int]:
        """Getter for `token_doc_freq`.
        Generate document frequency for tokens.
        """
        return {
            token: len(doc_ids)
            for token, doc_ids in self.__token_doc_freq.items()
        }

    @token_doc_freq.setter
    def token_doc_freq(self, arg) -> Dict[str, int]:
        pass

    @property
    def token_freq(self) -> Dict[str, int]:
        """Getter for `token_freq`.
        Generate token frequency from `token_entity_freq`.
        """
        return {
            token: sum(list(entity_freq.values()))
            for token, entity_freq in self.token_entity_freq.items()
        }

    @token_freq.setter
    def token_freq(self, arg) -> Dict[str, int]:
        pass

    @property
    def num_uniq_entities(self) -> int:
        """Getter for `num_uniq_entities`."""
        return len(self._entity_to_label)

    @num_uniq_entities.setter
    def num_uniq_entities(self, arg) -> Dict[str, int]:
        pass

    def _to_lower(self, token: str) -> str:
        """Lowercase token if specified."""
        if self.to_lower:
            return token.lower()
        return token

    def _add_token(self, token: str) -> None:
        """Helper function to add token in vocab and assign it's
        token index.

        Args:
            token: string
        """
        self.uniq_tokens.add(token)
        if len(self.uniq_tokens) > self.num_uniq_tokens:
            # new token added
            idx = self.num_uniq_tokens
            self._token_to_idx.update({token: idx})
            self._idx_to_token.update({idx: token})
            self.num_uniq_tokens += 1

    def _add_entities(self, entities: Set[str]) -> None:
        """Helper function to add entity and assign it's entity label.

        Args:
            entities: list of string entities.
        """

        # sort entities to assign consecutive indexes to `B-` and `I-` tags
        #   of entities
        entities = sorted(
            [
                entity
                for entity in entities
                if entity
                not in [
                    TokenEntityVocab.PAD_TOKEN_ENTITY_LABEL,
                    TokenEntityVocab.UNTAG_ENTITY_LABEL,
                ]
            ],
            key=lambda x: f"{x.split('-')[1]}-{x.split('-')[0]}",
        )
        for entity in entities:
            label = self.num_uniq_entities
            self._entity_to_label.update({entity: label})
            self._label_to_entity.update({label: entity})

    def fit(self, annotatedtuples: List[AnnotationTuple]) -> None:
        """Generate vocab, token_entity_freq, token_to_idx,
            token_to_entity, entity_to_label, label_to_entity mappings
            from list of annotated tokens.

        Note: every time fit is called, class instances/variables
            are initialized.

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
                    ...
                ]

        Returns:
            None
        """
        self._setup()
        entities: Set[str] = set()
        for annotatedtuple in annotatedtuples:
            annotatedtuple = annotatedtuple._replace(
                token=self._to_lower(token=annotatedtuple.token)
            )
            # checking if current token not in given list of ignore tokens
            if annotatedtuple.token not in self.ignore_tokens:
                self._add_token(token=annotatedtuple.token)
                self.token_entity_freq[annotatedtuple.token].update(
                    {annotatedtuple.entity: 1}
                )
                self.__token_doc_freq[annotatedtuple.token].add(
                    annotatedtuple.doc_id
                )
            entities.add(annotatedtuple.entity)
        self._add_entities(entities=entities)

    def token_to_idx(self, tokens: Union[str, List[str]]) -> List[int]:
        """Map given string or list of string tokens into
            list of associated integer token indexes.

        Note: indexes for unknown tokens(token not in vocab)
            are replaced by `unkown` token index.

        Args:
            tokens: str or list of string tokens

        Returns:
            idxs: list of associated integer token indexes
        """
        if isinstance(tokens, str):
            tokens = [tokens]
        validate_list(tokens, str)
        return [
            self._token_to_idx[self._to_lower(token=token)] for token in tokens
        ]

    def idx_to_token(self, idxs: Union[int, List[int]]) -> List[str]:
        """Map given integer or list of integer token indexes into
            list of associated string tokens.

        Note: indexes for unknown tokens(token not in vocab)
            are replaced by `unknown` token index.

        Args:
            idxs: integer or list of integer token indexes

        Returns:
            tokens: list of associated string tokens
        """
        if isinstance(idxs, int):
            idxs = [idxs]
        validate_list(idxs, int)
        return [self._idx_to_token[idx] for idx in idxs]

    def entity_to_label(self, entities: Union[str, List[str]]) -> List[int]:
        """Map given string or list of string entities into
            list of associated integer entity labels.

        Args:
            entities: string or list of string entities

        Returns:
            labels: list of associated integer entity labels
        """
        if isinstance(entities, str):
            entities = [entities]
        validate_list(entities, str)
        return [self._entity_to_label[entity] for entity in entities]

    def label_to_entity(self, labels: Union[int, List[int]]) -> List[str]:
        """Map given integer or list of integer entity labels into
            list of associated string entity labels.

        Args:
            entities: integer or list of integer entity labels

        Returns:
            labels: list of associated integer entity labels
        """
        if isinstance(labels, int):
            labels = [labels]
        validate_list(labels, int)
        return [self._label_to_entity[label] for label in labels]

    def annotation_to_longannotation(
        self, annotatedtuple: AnnotationTuple
    ) -> LongAnnotationTuple:
        """Convert AnnotationTuple to LongAnnotationTuple
            by mapping token index and entity label.

        Args:
            annotatedtuple: An annotated tuple
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
            token_idx=self.token_to_idx(annotatedtuple.token)[0],
            entity_label=self.entity_to_label(annotatedtuple.entity)[0],
        )()
