"""Contain general utility functions"""
from typing import Callable


def copy_docstring(original: Callable) -> Callable:
    """Copy docstring of one function to another."""

    def wrapper(target: Callable):
        target.__doc__ = original.__doc__
        return target

    return wrapper

