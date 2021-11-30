from copy import deepcopy
from typing import Any, Callable, List, Union

import pandas as pd


from ner_ehr.data.variables import AnnotationTuple, TokenTuple

from collections import Counter, defaultdict

from abc import ABC


class Vocab(ABC):
    """Creates a vocabulary for dataset"""

    def __init__(self):
        self.vocab = set()
        self.token_to_entity = defaultdict(Counter)
        self.token_doc_freq = defaultdict(set)

    def fit(self, annotated_tokens: List[AnnotationTuple]) -> None:
        """
        Adds the token and it's corresponding tags and counts in a dictionary of dictionary
            stats = {
                'token1': {'tag1': count, 'tag2': count},
                'token2': {'tag1': count}, ...}

        Args:
            token: string
            tag: string containing entity class of the token

        Returns:
            None
        """
        for token in annotated_tokens:
            self.vocab.add(token.token)
            self.token_to_entity[token.token].update({token.entity: 1})
            self.token_doc_freq[token.token].add(token.doc_id)


def sort_namedtuples(
    func: Callable[[Any], Union[List[AnnotationTuple], List[TokenTuple]]],
    by: Union[str, List[str]] = "start_idx",
):
    """Wrapper to sort list of AnnotationTuples or TokenTuples
    by start_idx in ascending order.

    Note: NamedTuples are sort in ascending order by start_idx
    """

    def wrapper(*args, **kwargs):
        namedtuples = func(*args, **kwargs)

        if "start_idx" not in namedtuples[0]._fields:
            raise AttributeError(
                f"Field start_idx missing in {type(namedtuples[0]).__name__}"
            )
        df = pd.DataFrame(namedtuples).sort_values(by=by)
        return list(
            df.itertuples(name=type(namedtuples[0]).__name__, index=False)
        )

    return wrapper


def df_to_namedtuples(
    name: str, df: pd.core.frame.DataFrame
) -> Union[List[AnnotationTuple], List[TokenTuple]]:
    """Convert given dataframe into list of namedtuples."""
    return list(df.itertuples(name=name, index=False))


def generate_annotated_token_seqs(
    annotatedtuples: List[AnnotationTuple], seq_length: int = 256
) -> List[List[AnnotationTuple]]:
    """Generate sequences of AnnotatedTuples of given seq_length.

        Note: Ensure that no sub-sequence can end with `I`-entity tag.
            Doing so a sub-sequence can be shorter than given length.
            Not all sequences are of given seq_length.

    Args:
        annotatedtuples: a list of AnnotatedTuples

        seq_length: maximum length of sub-sequences

    Returns:
        A list of list of AnnotatedTuples
    """

    seqs: List[List[AnnotationTuple]] = []
    total: int = len(annotatedtuples)

    def find_end(start: int):
        end = start + seq_length
        if end > total:
            end = total
        return end

    start: int = 0
    end: int = find_end(start)

    while end > start and end <= total:
        sub_sequences = deepcopy(annotatedtuples[start:end])
        is_I_found: bool = False
        while sub_sequences[-1].entity.startswith("I-"):
            sub_sequences.pop(-1)
            end -= 1
            is_I_found = True

        if is_I_found and sub_sequences:
            sub_sequences.pop(-1)  # removing `B-` entity
            end -= 1

        if sub_sequences:
            seqs.append(sub_sequences)
        start = end
        end = find_end(start)

    return seqs
