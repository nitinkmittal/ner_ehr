"""Contain general utility functions"""
from typing import Callable, List, Union


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
