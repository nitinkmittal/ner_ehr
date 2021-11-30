"""This module can be used to build CSVs with annotated
    token tuples from electronic health records."""
import os
from abc import ABC
from pathlib import Path
from typing import List, Union

import pandas as pd
from pandera import check_output
import pandera as pa
from pandera import Column, DataFrameSchema

from ner_ehr.data.variables import AnnotationTuple, TokenTuple

from ner_ehr.data import Constants


# schema for annotations  dataframe
annotations_df_schema = DataFrameSchema(
    {
        "doc_id": Column(pa.String, nullable=False),
        "token": Column(pa.String, nullable=False),
        "start_idx": Column(pa.Int, nullable=False),
        "end_idx": Column(pa.Int, nullable=False),
        "entity": Column(pa.String, nullable=False),
    }
)

# schema for tokens dataframe
tokens_df_schema = DataFrameSchema(
    {
        "doc_id": Column(pa.String, nullable=False),
        "token": Column(pa.String, nullable=False),
        "start_idx": Column(pa.Int, nullable=False),
        "end_idx": Column(pa.Int, nullable=False),
    }
)


def read_csv(fp: Union[Path, str], **kwargs) -> pd.core.frame.DataFrame:
    return pd.read_csv(fp, **kwargs)


class CoNLLDataset(ABC):
    """Save list of annotated/unannoted tokens in CSV."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.tokens_with_annotations: pd.core.frame.DataFrame = None
        self.tokens_without_annotations: pd.core.frame.DataFrame = None

    def write_csv_tokens_with_annotations(self, **kwargs) -> None:
        raise NotImplementedError

    def write_csv_tokens_without_annotations(self, **kwargs) -> None:
        raise NotImplementedError

    @check_output(annotations_df_schema)
    def read_csv_tokens_with_annotations(
        self, **kwargs
    ) -> pd.core.frame.DataFrame:
        raise NotImplementedError

    @check_output(tokens_df_schema)
    def read_csv_tokens_without_annotations(
        self, **kwargs
    ) -> pd.core.frame.DataFrame:
        raise NotImplementedError


class EHR(CoNLLDataset):
    """Save tokens and annotations for EHR into CoLNNDataset format"""

    def __init__(
        self,
    ):
        super().__init__()

    def _write_csv(self, fp: Union[str, Path], is_annotated: bool) -> None:
        """Write dataframe into CSV.

        Args:
            fp: filepath with file name to be used
                while saving CSV

            is_annotated: boolean flag to indicate token with
                or without annotations.
        """
        filename = os.path.basename(fp).split(".")[0]
        add_on = (
            "-tokens-with-annotations.csv"
            if is_annotated
            else "-tokens-without-annotations.csv"
        )
        fp = os.path.join(os.path.dirname(fp), f"{filename}{add_on}")

        if is_annotated:
            self.tokens_with_annotations.to_csv(fp, index=False)
        else:
            self.tokens_without_annotations.to_csv(fp, index=False)

    def write_csv_tokens_with_annotations(
        self,
        tokens: List[TokenTuple],
        annotations: List[AnnotationTuple],
        fp: Union[str, Path],
    ) -> None:
        """Save tokens with annotations into CSV."""

        self.tokens_with_annotations = pd.merge(
            pd.DataFrame(tokens),
            pd.DataFrame(annotations),
            on=tokens[0]._fields,
            how="left",
        )
        self.tokens_with_annotations["entity"].fillna(
            Constants.UNTAG_ENTITY_LABEL, inplace=True
        )
        annotations_df_schema.validate(self.tokens_with_annotations)
        self._write_csv(fp=fp, is_annotated=True)

    def write_csv_tokens_without_annotations(
        self,
        tokens: List[TokenTuple],
        fp: Union[str, Path],
    ) -> None:
        """Save tokens without annotations into CSV."""
        self.tokens_without_annotations = pd.DataFrame(tokens)
        tokens_df_schema.validate(self.tokens_without_annotations)
        self._write_csv(fp=fp, is_annotated=False)

    @check_output(annotations_df_schema)
    def read_csv_tokens_with_annotations(
        self, fp: Union[Path, str]
    ) -> pd.core.frame.DataFrame:
        return read_csv(
            fp=fp,
            converters={
                "doc_id": str,
                "token": str,
                "start_idx": int,
                "end_idx": int,
                "entity": str,
            },
        )

    @check_output(tokens_df_schema)
    def read_csv_tokens_without_annotations(
        self, fp: Union[Path, str]
    ) -> pd.core.frame.DataFrame:
        return read_csv(
            fp=fp,
            converters={
                "doc_id": str,
                "token": str,
                "start_idx": int,
                "end_idx": int,
            },
        )
