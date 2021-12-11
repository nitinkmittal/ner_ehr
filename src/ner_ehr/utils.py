"""This module contains general utility functions for this package."""
import os
from glob import glob
from pathlib import Path
from time import time
from typing import Any, Callable, List, Union

import numpy as np
import yaml

from ner_ehr.data.ehr import EHR
from ner_ehr.data.utils import df_to_namedtuples
from ner_ehr.data.variables import AnnotationTuple


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
        annotatedtuples += df_to_namedtuples(
            name=AnnotationTuple.__name__,
            df=EHR.read_csv_tokens_with_annotations(fp),
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
