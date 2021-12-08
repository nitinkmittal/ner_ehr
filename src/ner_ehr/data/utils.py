"""This module contain helper utilities to prepare data for NER."""
from pathlib import Path
from typing import List, Union

import pandas as pd

from ner_ehr.data.variables import AnnotationTuple, TokenTuple

# def sort_namedtuples(
#     func: Callable[[Any], Union[List[AnnotationTuple], List[TokenTuple]]],
#     by: Union[str, List[str]] = ["doc_id", "start_idx"],
# ):
#     """Wrapper to sort list of AnnotationTuples or TokenTuples
#     by `start_idx` in ascending order.

#     Note: NamedTuples are sort in ascending order by doc_id, start_idx
#     """

#     def wrapper(*args, **kwargs):
#         namedtuples = func(*args, **kwargs)

#         if "doc_id" not in namedtuples[0]._fields:
#             raise AttributeError(
#                 f"Field `doc_id` missing in {type(namedtuples[0]).__name__}"
#             )

#         if "start_idx" not in namedtuples[0]._fields:
#             raise AttributeError(
#                 f"Field `start_idx` missing in {type(namedtuples[0]).__name__}"
#             )
#         df = pd.DataFrame(namedtuples).sort_values(by=by)
#         return list(
#             df.itertuples(name=type(namedtuples[0]).__name__, index=False)
#         )

#     return wrapper


def sort_namedtuples(
    namedtuples: Union[List[AnnotationTuple], List[TokenTuple]],
    by: Union[str, List[str]] = ["doc_id", "start_idx"],
    ascending: bool = True,
):
    """Sort list of TokenTuples/AnnotationTuples/LongAnnotationTuples
    by given condition in given order.

    Note: NamedTuples are sort in ascending order by doc_id, start_idx
    """

    def check_field(field: str):
        if field not in namedtuples[0]._fields:
            raise AttributeError(
                f"Field `{field}` missing in `{type(namedtuples[0]).__name__}`"
            )

    if isinstance(by, str):
        by = [by]
    for field in by:
        check_field(field=field)

    df = pd.DataFrame(namedtuples).sort_values(by=by, ascending=ascending)
    return list(df.itertuples(name=type(namedtuples[0]).__name__, index=False))


def df_to_namedtuples(
    name: str, df: pd.core.frame.DataFrame
) -> Union[List[AnnotationTuple], List[TokenTuple]]:
    """Convert given dataframe into list of namedtuples."""
    return list(df.itertuples(name=name, index=False))


def generate_token_seqs(
    annotatedtuples: List[AnnotationTuple], seq_length: int = 256
) -> List[List[AnnotationTuple]]:
    """Generate sequences of AnnotatedTuples of given seq_length.

        Note: Ensure that sequence of tokens generated preserve
            continuation of `IOB` tagging within every sequence,
            i.e. `B-` and `I-` entity tags in continuation
                are kept in same sequence.
            Doing so, a sequence can be shorter than given length,
            i.e. not all sequences are of given `seq_length`.

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

        seq_length: maximum length of sub-sequences

    Returns:
        A list of list of AnnotatedTuples
    """

    if seq_length == 1:
        raise ValueError(
            "Sequence generation not implemented for seq_length = 1"
        )

    seqs: List[List[AnnotationTuple]] = []
    total_tokens: int = len(annotatedtuples)

    def find_end(start: int):
        """Find end index for a sequence  with respect to start idx."""
        end = start + seq_length
        if end > total_tokens:
            end = total_tokens
        return end

    start: int = 0
    end: int = find_end(start)

    while end > start and end <= total_tokens:
        if (
            not annotatedtuples[start].entity.startswith("I-")
            and annotatedtuples[end - 1].entity.startswith("I-")
            and (
                end == total_tokens
                or not annotatedtuples[end].entity.startswith("I-")
            )
        ):
            # perfect sequence found
            # TODO: Add examples of perfect sequences
            pass
        else:
            # finding best possible sequence
            is_inside_tag_found: bool = False
            # check if inside tag and not retracking to previous sub-seqs
            while end > start and annotatedtuples[end - 1].entity.startswith(
                "I-"
            ):
                end -= 1
                is_inside_tag_found = True

            if is_inside_tag_found:
                end -= 1

        if end <= start:
            # skipping current sequence
            #   try from next start idx
            start += 1
            end = find_end(start)
            continue

        seqs.append(annotatedtuples[start:end])
        start = end
        end = find_end(start)
    return seqs


def read_csv(fp: Union[Path, str], **kwargs) -> pd.core.frame.DataFrame:
    """Helper function to read a CSV."""
    return pd.read_csv(fp, **kwargs)
