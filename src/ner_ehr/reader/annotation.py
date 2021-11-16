"""This module contains methods to read raw EHR data from file."""
from pathlib import Path
from typing import Union
from .callbacks.annotation import Parser, Validator
from .base import ABC, BaseReader


class Reader(BaseReader):
    def __init__(self, parser: Parser, validator: Validator):
        super().__init__(parser=parser, validator=validator)

    def read(self, fp: Union[Path, str]):
        """Function to read and return text from file.

        Args:
            fp: file path

        Returns:
            text from given file path
        """
        annotations = []
        with open(fp, "r") as f:
            for line in f.readlines():
                if self.validator(line):
                    line = self.parser(line)
                    annotations += line
        return annotations


class Annotation(ABC):
    def __init__(
        self, entity_type: str, start_idx: int, end_idx: int, entity_name: str
    ):
        self.entity_type = entity_type
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.entity_name = entity_name

    @property
    def start_idx(self):
        return self._start_idx

    @start_idx.setter
    def start_idx(self, idx):
        self._start_idx = int(idx)

    @property
    def end_idx(self):
        return self._end_idx

    @end_idx.setter
    def end_idx(self, idx):
        self._end_idx = int(idx)

    def get(
        self,
    ):
        return (
            self.entity_type,
            self.start_idx,
            self.end_idx,
            self.entity_name,
        )

    def __repr__(
        self,
    ):
        return (
            f"Entity: {self.entity_type}, ",
            f"start_idx: {self.start_idx}, ",
            f"end_idx: {self.end_idx}, ",
            f"entity_name: {self.entity_name}",
        )
