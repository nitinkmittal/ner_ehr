"""Contain utilities to handle embeddings."""
from abc import ABC
from pathlib import Path
from typing import List, Union

import numpy as np
from gensim.models import KeyedVectors, keyedvectors

from ner_ehr.utils import copy_docstring, validate_list


class Embeddings(ABC):
    """Base class for embeddings.

    Methods under this class can be used to load embedding
    in gensim.word2vec format
    """

    def __init__(self, unknown_token_embedding: np.ndarray, **kwargs):
        """
        Args:
            unknown_token_embedding: embedding vector for tokens not
                present in pre-trained embeddings. Dimension of unknown
                embedding vector should be equal to dimension of pre-trained
                embedding vectors

            **kwargs: other keyword-arguments
        """
        self.embeddings: keyedvectors.KeyedVectors = None
        self.unknown_token_embedding = unknown_token_embedding
        self.__dict__.update(kwargs)

    def load_word2vec(self, **kwargs) -> None:
        """Helper function to load embeddings in form of genism word2vec."""
        raise NotImplementedError("Not implemented ..")

    def __call__(
        self, tokens: Union[str, List[str]], to_lower: bool = True
    ) -> np.ndarray:
        """Return NumPy array containing embedding vectors for given tokens

        `unknown_token_embedding` is used for tokens not present
        in pre-trained word2vec embeddings.

        Args:
            tokens: a list of string tokens

            to_lower: boolean flag, default=True
                if True, tokens are lowercased otherwise not

        Returns:
            A NumPy array with pre-trained embeddings
        """
        validate_list(l=tokens, dtype=str)
        embeddings = []
        for token in tokens:
            token = token.lower() if to_lower else token
            embeddings.append(
                self.embeddings.get_vector(token)
                if token in self.embeddings.key_to_index
                else self.unknown_token_embedding
            )

        return np.array(embeddings)


class GloveEmbeddings(Embeddings):
    def __init__(
        self, unknown_token_embedding: np.ndarray, glove_fp: Union[str, Path]
    ):
        super().__init__(
            unknown_token_embedding=unknown_token_embedding, glove_fp=glove_fp
        )

    @copy_docstring(Embeddings.load_word2vec)
    def load_word2vec(
        self,
        force_load: bool = False,
    ):
        """Note: embeddings are for uncased tokens."""
        if (
            isinstance(self.embeddings, keyedvectors.KeyedVectors)
            and not force_load
        ):
            return

        self.embeddings = KeyedVectors.load_word2vec_format(
            self.glove_fp,
            binary=False,
            no_header=True,
        )
        if len(self.unknown_token_embedding) != self.embeddings.vector_size:
            raise ValueError(
                "Vector size of `unknown_token_embedding`: "
                f"{self.unknown_token_embedding.shape} "
                f"!= glove embedding-size: ({self.embeddings.vector_size},)"
            )


class PubMedicalEmbeddings(Embeddings):
    def __init__(
        self, unknown_token_embedding: np.ndarray, pubmed_fp: Union[str, Path]
    ):
        super().__init__(
            unknown_token_embedding=unknown_token_embedding,
            pubmed_fp=pubmed_fp,
        )

    @copy_docstring(Embeddings.load_word2vec)
    def load_word2vec(
        self,
        force_load: bool = False,
    ):
        """Note: embeddings are for uncased tokens."""
        if (
            isinstance(self.embeddings, keyedvectors.KeyedVectors)
            and not force_load
        ):
            return

        self.embeddings = KeyedVectors.load_word2vec_format(
            self.pubmed_fp,
            binary=True,
            no_header=False,
        )
        if len(self.unknown_token_embedding) != self.embeddings.vector_size:
            raise ValueError(
                "Vector size of `unknown_token_embedding`: "
                f"{self.unknown_token_embedding.shape} "
                f"!= glove embedding-size: ({self.embeddings.vector_size},)"
            )
