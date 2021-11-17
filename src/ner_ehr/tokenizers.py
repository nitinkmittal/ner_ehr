"""This module contains defination for base tokenizer."""
from abc import ABC
from collections import namedtuple
from typing import Callable, List

from ner_ehr.utils import copy_docstring

from ner_ehr.data.entities import Token, TokenTuple


def _validate_token_idxs(tokens: List[TokenTuple], text: str) -> None:
    """Validate assigned indexes for each token from given text.

    Raises ValueError if incorrect indexes are assigned to tokens.

    Args:
        tokens: a list of Token(token, start_idx, end_idx)

        text: string from which tokens are generated from

    Returns:
        None
    """
    for token in tokens:
        if token.token != text[token.start_idx : token.end_idx]:
            raise ValueError(
                f"Incorrect indexes: ({token.start_idx}, {token.end_idx}) "
                f"mapped for token: {token.token}, "
                f"'{token.token}' != '{text[token.start_idx : token.end_idx]}'"
            )


class Tokenizer(ABC):
    """Base tokenizer."""

    def __init__(
        self,
        tokenizer: Callable[[str], List[str]],
        validate_token_idxs: bool = False,
    ):
        """
        Args:
            tokenizer (callable): to convert string into list of tokens
            >>> tokenizer.tokenize('this is a sentence') -> ['this', 'is', 'a', 'sentence']

            validate_token_idxs (boolean): if True, start and end character
                indexes for each token are validated for given text, default=False
        """
        self.tokenizer = tokenizer
        self.validate_token_idxs = validate_token_idxs

    def tokenize(self, text: str) -> List[str]:
        """Convert given string into list of string tokens.

        Args:
            text: a string

        Returns:
            A list of string tokens
        """
        pass

    def _map_token_idxs(
        self,
        tokens: List[str],
        text: str,
    ) -> List[TokenTuple]:
        """Map start and end character indexes
            for each token generated from given text.

        Args:
            tokens: a list of string tokens

            text: a string from which tokens are generated

        Returns:
            A list of NamedTuples, Tokens(token, start_idx, end_idx)
        """
        idx_mapped_tokens = []
        start_idx, end_idx = 0, 0
        for raw_token in tokens:
            raw_token = str(raw_token).strip()
            if not raw_token:  # skip if empty token
                continue

            # find index of first character of token
            while raw_token[0] != text[start_idx]:
                start_idx += 1

            # find index of last character of token
            end_idx = start_idx + len(raw_token)
            idx_mapped_tokens.append(
                Token(token=raw_token, start_idx=start_idx, end_idx=end_idx)()
            )

            start_idx += len(raw_token)  # update start_idx for next token

        if self.validate_token_idxs:
            _validate_token_idxs(tokens=idx_mapped_tokens, text=text)
        return idx_mapped_tokens

    def __call__(self, text: str) -> List[TokenTuple]:
        tokens = self.tokenize(text)
        return self._map_token_idxs(tokens=tokens, text=text)


class SplitTokenizer(Tokenizer):
    """Split given string into list of string tokens at given separator."""

    def __init__(self, sep: str = " ", validate_token_idxs: bool = False):
        """
        Args:
            sep: separator used to split a string into list of string tokens

            validate_token_idxs (boolean): if True, start and end character
                indexes for each token are validated for given text,
                default=False
        """
        tokenizer = lambda x: x.split(sep)
        super().__init__(
            tokenizer=tokenizer,
            validate_token_idxs=validate_token_idxs,
        )

    @copy_docstring(Tokenizer.tokenize)
    def tokenize(self, text: str) -> List[str]:
        return self.tokenizer(text)


class ScispacyTokenizer(Tokenizer):
    """Scispacy tokenizer from a spaCy NER model
    trained on the BC5CDR corpus."""

    def __init__(self, validate_token_idxs: bool = False):
        """
        Args:
            sep: separator used to split a string into list of string tokens

            validate_token_idxs (boolean): if True, start and end character
                indexes for each token are validated for given text,
                default=False
        """
        import en_ner_bc5cdr_md

        super().__init__(
            tokenizer=en_ner_bc5cdr_md.load().tokenizer,
            validate_token_idxs=validate_token_idxs,
        )

    @copy_docstring(Tokenizer.tokenize)
    def tokenize(self, text: str) -> List[str]:
        return self.tokenizer(text)
