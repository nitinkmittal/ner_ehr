"""This module contain utilities to read pre-trained embeddings
    into `gensim.models.keyedvectors`."""
from abc import ABC
from pathlib import Path
from typing import List, Union

import numpy as np
from gensim.models import KeyedVectors, keyedvectors

from ner_ehr.utils import validate_list

TO_LOWER: bool = False

FORCE_LOAD = False


class Embeddings(ABC):
    """Base class to read pre-trained embeddings.

    Methods under this class can be used to load pre-trained embeddings
    in `gensim.models.keyedvectors` format.
    """

    def __init__(
        self,
        unknown_token_embedding: np.ndarray,
        to_lower: bool = TO_LOWER,
        **kwargs,
    ):
        """
        Args:
            unknown_token_embedding: embedding vector to be used
                for tokens not present in pre-trained embeddings vocab.
                Dimension of unknown embedding vector should be equal
                to dimension of pre-trained embedding vectors

            to_lower: boolean flag, default=True
                if True, tokens are lowercased otherwise not

            **kwargs: other keyword-arguments
        """
        self.to_lower = to_lower
        self.unknown_token_embedding = unknown_token_embedding
        self.embeddings: keyedvectors.KeyedVectors = None
        self.__dict__.update(kwargs)

    def load_word2vec(self, **kwargs) -> None:
        """Helper function to load pre-trained embeddings in
        `gensim.models.keyedvectors` format."""
        raise NotImplementedError("Not implemented ..")

    def __call__(self, tokens: Union[str, List[str]]) -> np.ndarray:
        """Return NumPy array containing embedding vectors for given token/s.

        `unknown_token_embedding` is used for tokens not present
        in pre-trained embeddings vocab.

        Args:
            tokens: string or list of string tokens

        Returns:
            embeddings: A 2-D NumPy array of shape
                (number of tokens, embedding dimension)
                containing pre-trained embeddings for given tokens
        """
        if isinstance(tokens, str):
            tokens = [tokens]

        validate_list(l=tokens, dtype=str)
        embeddings = []
        for token in tokens:
            token = token.lower() if self.to_lower else token
            embeddings.append(
                self.embeddings.get_vector(token)
                if token in self.embeddings.key_to_index
                else self.unknown_token_embedding
            )

        return np.array(embeddings)


class GloveEmbeddings(Embeddings):
    """Load pre-trained `glove` embeddings in `gensim.models.keyedvectors`
    format."""

    def __init__(
        self,
        unknown_token_embedding: np.ndarray,
        glove_fp: Union[str, Path],
        to_lower: bool = TO_LOWER,
    ):
        """
        Args:
            unknown_token_embedding: embedding vector to be used
                for tokens not present in pre-trained embeddings vocab.
                Dimension of unknown embedding vector should be equal
                to dimension of pre-trained embedding vectors

            glove_fp: filepath to pre-trained glove embeddings

            to_lower: boolean flag, default=True
                if True, tokens are lowercased otherwise not
        """
        super().__init__(
            unknown_token_embedding=unknown_token_embedding,
            to_lower=to_lower,
            glove_fp=glove_fp,
        )

    def load_word2vec(
        self,
        force_load: bool = FORCE_LOAD,
    ):
        """Load pre-trained `glove` embeddings in
        `gensim.models.keyedvectors` format.

        Pre-trained can be found on https://nlp.stanford.edu/projects/glove/

        Args:
            force_load: boolean flag, if True embeddings are loaded everytime
                this function is called, otherwise not, default=False
        """
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
    """Load pre-trained `PubMedical` embeddings in `gensim.models.keyedvectors`
    format."""

    def __init__(
        self,
        unknown_token_embedding: np.ndarray,
        pubmed_fp: Union[str, Path],
        to_lower: bool = TO_LOWER,
    ):
        """
        Args:
            unknown_token_embedding: embedding vector to be used
                for tokens not present in pre-trained embeddings vocab.
                Dimension of unknown embedding vector should be equal
                to dimension of pre-trained embedding vectors

            glove_fp: filepath to pre-trained PubMedical embeddings

            to_lower: boolean flag, default=True
                if True, tokens are lowercased otherwise not
        """
        super().__init__(
            unknown_token_embedding=unknown_token_embedding,
            to_lower=to_lower,
            pubmed_fp=pubmed_fp,
        )

    def load_word2vec(
        self,
        force_load: bool = FORCE_LOAD,
    ):
        """Load pre-trained `Biomedical Public` embeddings in
        `gensim.models.keyedvectors` format.

        Pre-trained `Biomedical Public` can be found on
        https://archive.org/download/pubmed2018_w2v_200D.tar/pubmed2018_w2v_200D.tar.gz

        force_load: boolean flag, if True embeddings are loaded everytime
                this function is called, otherwise not, default=False
        """
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
                f"!= `pubmed` embedding-size: ({self.embeddings.vector_size},)"
            )
