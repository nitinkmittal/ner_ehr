from copy import deepcopy
from typing import Any, Callable, List, Union

import pandas as pd


from ner_ehr.data.variables import AnnotationTuple, TokenTuple


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


def split_annotated_tokens_in_batches(
    annotatedtuples: List[AnnotationTuple], seq_length: int = 256
) -> List[List[AnnotationTuple]]:
    """Split list of AnnotatedTuples into batches of given seq_length.

    Args:
        annotatedtuples: a list of AnnotatedTuples

        seq_length: maximum length of sub sequences

    Returns:
        A list of list of AnnotatedTuples
    """

    batches: List[List[AnnotationTuple]] = []
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
        while sub_sequences[-1].entity.startswith("I-"):
            sub_sequences.pop(-1)
            end -= 1
        if sub_sequences:
            batches.append(sub_sequences)
        start = end
        end = find_end(start)

    return batches
