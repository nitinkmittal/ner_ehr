"""This module contain helper utilities to prepare data for NER."""
from typing import List, Union

import pandas as pd

import warnings
from ner_ehr.data.variables import (
    AnnotationTuple,
    LongAnnotationTuple,
    TokenTuple,
)


def sort_namedtuples(
    namedtuples: Union[
        List[TokenTuple], List[AnnotationTuple], List[LongAnnotationTuple]
    ],
    by: Union[str, List[str]] = ["doc_id", "start_idx"],
    ascending: bool = True,
    drop_duplicates: bool = False,
    keep: str = "first",
):
    """Sort list of TokenTuples/AnnotationTuples/LongAnnotationTuples
    by given condition in given order.

    Args:
        namedtuples: a list of
            tokentuples, ex: [
                                Token(
                                    doc_id="100035",
                                    token='recurrent',
                                    start_idx=10179,
                                    end_idx=10188,),
                                ...
                            ]
            or
            annotatedtuples, ex: [
                                Annotation(
                                    doc_id="100035",
                                    token='recurrent',
                                    start_idx=10179,
                                    end_idx=10188,
                                    entity='B-Reason',),
                                ...
                            ]
            or
            long_annotationtuples, ex: [
                                LongAnnotation(
                                    doc_id="100035",
                                    token='recurrent',
                                    start_idx=10179,
                                    end_idx=10188,
                                    entity='B-Reason',
                                    token_idx=2,
                                    entity_label=2,),
                                ...
                            ]

        by: string or list of string,
            fields within each namedtuple

        ascending: boolean flag, default=False,
            if True, namedtuples are sorted in ascending order on
            given condition other sorted in descending order

        drop_duplicated: boolean flag, default=False

        keep (str): {‘first’, ‘last’, False}, default ‘first’
            Determines which duplicates (if any) to keep.
            - first: Drop duplicates except for the first occurrence.
            - last: Drop duplicates except for the last occurrence.
            - False: Drop all duplicates.

        Returns:
            A list of sorted namedtuples
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

    df = pd.DataFrame(namedtuples)
    if not drop_duplicates and df.duplicated(subset=by).sum() > 0:
        warnings.simplefilter("always", UserWarning)
        warnings.warn(
            f"Duplicates found on given sort condition "
            f"[{', '.join(by)}]. "
            "May lead to incorrect sorting, please check sorted namedtuples."
        )
    else:
        df = df.drop_duplicates(subset=by, keep=keep).reset_index(drop=True)

    df = df.sort_values(by=by, ascending=ascending)
    return list(df.itertuples(name=type(namedtuples[0]).__name__, index=False))


def df_to_namedtuples(
    name: str, df: pd.core.frame.DataFrame
) -> Union[List[TokenTuple], List[AnnotationTuple], List[LongAnnotationTuple]]:
    """Convert given dataframe into list of namedtuples.

    Args:
        name: string name of every namedtuple formed from rows of dataframe.

    Returns:
        A list of namedtuples
    """
    return list(df.itertuples(name=name, index=False))


def generate_token_seqs(
    annotatedtuples: Union[List[AnnotationTuple], List[LongAnnotationTuple]],
    seq_length: int = 256,
) -> Union[List[List[AnnotationTuple]], List[List[LongAnnotationTuple]]]:
    """Generate sequences of AnnotatedTuples of with maximum length
        as given `seq_length`.

    Note: this function ensures that sequences of tokens generated preserve
        continuation of `IOB` tagging within every sequence,
        i.e. `B-` and `I-` tags of an entity in continuation
        are kept in same sequence.
        Doing so, a sequence can be shorter than given `seq_length`,
        i.e. not all sequences are of given `seq_length`.

        Also, this function is not implemented to generated sequences
            of `seq_length=1`

    Args:
        namedtuples: a list of
            annotatedtuples, ex: [
                                    Annotation(
                                        doc_id="100035",
                                        token='recurrent',
                                        start_idx=10179,
                                        end_idx=10188,
                                        entity='B-Reason',),
                                    ...
                                    ]
            or
            long_annotationtuples, ex: [
                                    LongAnnotation(
                                        doc_id="100035",
                                        token='recurrent',
                                        start_idx=10179,
                                        end_idx=10188,
                                        entity='B-Reason',
                                        token_idx=2,
                                        entity_label=2,),
                                    ...
                                    ]

        seq_length: maximum length of sequences, default=256

    Returns:
        A list of list of AnnotatedTuples
    """

    if seq_length == 1:
        raise ValueError("Not implemented for seq_length=1")

    seqs: Union[
        List[List[AnnotationTuple]], List[List[LongAnnotationTuple]]
    ] = []
    total_tokens: int = len(annotatedtuples)

    def find_end(start: int):
        """Find end index for a sequence with respect to start idx."""
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
