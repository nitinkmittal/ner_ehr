"""This module contains general utility functions for this package."""
import os
from glob import glob
from pathlib import Path
from time import time
from typing import Any, Callable, List, Union, Tuple, NamedTuple

import numpy as np
import yaml

from ner_ehr.data.ehr import EHR
from ner_ehr.data.utils import df_to_namedtuples
from ner_ehr.data.variables import AnnotationTuple

from ner_ehr.data import Constants

from ner_ehr.data.utils import sort_namedtuples


def validate_list(l: List[Union[int, str]], dtype: type):
    """Helper function to validate given type of input and it's value."""
    if not isinstance(l, list):
        raise TypeError(f"Expect input to be a list, not {type(l)}")

    if not all(isinstance(item, dtype) for item in l):
        raise TypeError(f"Expected dtype of all items in list to be {dtype}")


def copy_docstring(original: Callable) -> Callable:
    """Copy docstring of one function to another."""

    def wrapper(target: Callable):
        target.__doc__ = original.__doc__
        return target

    return wrapper


def save_np(arr: np.ndarray, fp: Union[str, Path]) -> None:
    """Save given array as .npy file."""
    with open(fp, "wb") as f:
        np.save(f, arr)


def load_np(fp: Union[str, Path]) -> np.ndarray:
    """Load array from given .npy file."""
    with open(fp, "rb") as f:
        arr = np.load(f)
    return arr


def save_kwargs(fp: Union[str, Path], **kwargs) -> None:
    """Save keyword-arguments in a YAML file."""
    with open(fp, "w") as f:
        yaml.dump(kwargs, f, sort_keys=False)


def read_annotatedtuples(dir: Union[str, Path]) -> List[AnnotationTuple]:
    """Read annotated tuples from CSVs present inside given directory.

    Args:
        dir: directory containing CSVs with annotated tokens

    Returns:
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
    annotatedtuples = []
    for fp in glob(os.path.join(dir, r"*.csv")):
        df = EHR.read_csv_tokens_with_annotations(fp)
        df = df.drop_duplicates(
            subset=["doc_id", "token", "start_idx", "end_idx"], keep="last"
        )
        df.reset_index(drop=True)
        annotatedtuples += df_to_namedtuples(
            name=AnnotationTuple.__name__,
            df=df,
        )

    return annotatedtuples


def time_func(func: Callable[[Any], Any]):
    """Wrapper to time given function."""

    def wrapper(*args, **kwargs):
        start = time()
        output = func(*args, **kwargs)
        print(f"{func.__name__} took {time()-start:.5f} seconds")
        return output

    return wrapper


def compare_namedtuples(a: NamedTuple, b: NamedTuple, fields: List[str]):
    """Raise `ValueError` if given field does not match in a and b."""
    for field in fields:
        if getattr(a, field) != getattr(b, field):
            raise ValueError(f"Field `{field}` not matching in {a} and {b}")


def aggregrate_true_pred_tokens(
    true: List[AnnotationTuple], pred: List[AnnotationTuple]
) -> Tuple[List[AnnotationTuple], List[AnnotationTuple]]:
    """Combine `IOB` entity tagging."""

    assert len(true) == len(pred)
    true = sort_namedtuples(true, by=["doc_id", "start_idx"], ascending=True)
    pred = sort_namedtuples(pred, by=["doc_id", "start_idx"], ascending=True)

    true_agg: List[AnnotationTuple] = []
    pred_agg: List[AnnotationTuple] = []

    extract_entity = (
        lambda entity: entity.split("-")[1] if "-" in entity else entity
    )

    i = 0
    while i < len(true):
        t, p = true[i], pred[i]
        compare_namedtuples(
            t, p, fields=["doc_id", "token", "start_idx", "end_idx"]
        )
        if t.entity == Constants.UNTAG_ENTITY_LABEL.value:
            t = t._replace(entity=[extract_entity(t.entity)])
            p = p._replace(entity=[extract_entity(p.entity)])
        elif "B-" in t.entity:
            doc_id = t.doc_id
            token = t.token
            start_idx = t.start_idx
            end_idx = t.end_idx

            t_entity = extract_entity(t.entity)
            p_entity = extract_entity(p.entity)
            t_entities = [t_entity]
            p_entities = [p_entity]

            while (
                i < len(true) - 1
                and doc_id == true[i + 1].doc_id
                and f"I-{t_entity}" == true[i + 1].entity
            ):
                compare_namedtuples(
                    true[i + 1],
                    pred[i + 1],
                    fields=["doc_id", "token", "start_idx", "end_idx"],
                )
                token += " " + true[i + 1].token
                end_idx = true[i + 1].end_idx

                p_entity = extract_entity(pred[i + 1].entity)
                p_entities.append(p_entity)

                t_entity = extract_entity(true[i + 1].entity)
                t_entities.append(t_entity)
                i += 1

            t = AnnotationTuple(
                doc_id=doc_id,
                token=token,
                start_idx=start_idx,
                end_idx=end_idx,
                entity=t_entities,
            )
            p = AnnotationTuple(
                doc_id=doc_id,
                token=token,
                start_idx=start_idx,
                end_idx=end_idx,
                entity=p_entities,
            )

        true_agg.append(t)
        pred_agg.append(p)
        i += 1

    return true_agg, pred_agg
