"""This module can be used to read and write CSVs
    with annotated and unannotated token tuples."""
import os
from abc import ABC
from pathlib import Path
from typing import List, Union

import pandas as pd
import pandera as pa
from pandera import Column, DataFrameSchema, check_output

from ner_ehr.data import Constants
from ner_ehr.data.variables import AnnotationTuple, TokenTuple

# schema used to validate pandas dataframe columns and their data
df_schema = {
    "doc_id": Column(pa.String, nullable=False),
    "token": Column(pa.String, nullable=False),
    "start_idx": Column(pa.Int, nullable=False),
    "end_idx": Column(pa.Int, nullable=False),
    "entity": Column(pa.String, nullable=False),
}


# schema for annotations dataframe
annotations_df_schema = DataFrameSchema(
    {field: df_schema[field] for field in AnnotationTuple._fields}
)

# TODO: strict column checking for while reading unannotated tokens
# schema for tokens dataframe
tokens_df_schema = DataFrameSchema(
    {field: df_schema[field] for field in TokenTuple._fields}
)

# converters used to correctly interpret column datatypes
#   while reading data from CSVs
col_converters = {
    "doc_id": str,
    "token": str,
    "start_idx": int,
    "end_idx": int,
    "entity": str,
}

# column dtype converters for annotations dataframe
annotations_col_converters = {
    field: col_converters[field] for field in AnnotationTuple._fields
}


# column dtype converters for tokens dataframe
tokens_col_converters = {
    field: col_converters[field] for field in TokenTuple._fields
}


class EHR(ABC):
    """Save tokens and annotations for EHR data into CoNLLDataset format."""

    def __init__(
        self,
    ):
        self.tokens_with_annotations: pd.core.frame.DataFrame = None
        self.tokens_without_annotations: pd.core.frame.DataFrame = None

    def _write_csv(self, fp: Union[str, Path], is_annotated: bool) -> None:
        """Write dataframe with tokens (annotated/unannotated) into a CSV file.

        Args:
            fp: filepath use while saving CSV

            is_annotated: boolean flag to indicate token with
                or without annotations.

        Returns:
            None
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
        """Write dataframe with annotated tokens into a CSV file.

        Args:
            tokens: list of unannotated tokens
                Ex: [
                        Token(
                            doc_id="100035",
                            token='recurrent',
                            start_idx=10179,
                            end_idx=10188,),
                        Token(
                            doc_id="100035",
                            token='seizures',
                            start_idx=10189,
                            end_idx=10197,),
                        ...
                    ]

            annotations: list of annotated tokens
                Ex: [
                        Annotation(
                            doc_id="100035",
                            token='recurrent',
                            start_idx=10179,
                            end_idx=10188,
                            entity='B-Reason'),
                        Annotation(
                            doc_id="100035",
                            token='seizures',
                            start_idx=10189,
                            end_idx=10197,
                            entity='I-Reason'),
                        ...
                    ]
            fp: filepath use while saving CSV

        Returns:
            None
        """
        self.tokens_with_annotations = pd.merge(
            pd.DataFrame(tokens),
            pd.DataFrame(annotations),
            on=tokens[0]._fields,
            how="left",
        )
        # adding untag entity for unannotated tokens
        self.tokens_with_annotations["entity"].fillna(
            Constants.UNTAG_ENTITY_LABEL.value, inplace=True
        )
        annotations_df_schema.validate(self.tokens_with_annotations)
        self._write_csv(fp=fp, is_annotated=True)

    def write_csv_tokens_without_annotations(
        self,
        tokens: List[TokenTuple],
        fp: Union[str, Path],
    ) -> None:
        """Write dataframe with annotated tokens into a CSV file.

        Args:
            tokens: list of unannotated tokens
                Ex: [
                        Token(
                            doc_id="100035",
                            token='recurrent',
                            start_idx=10179,
                            end_idx=10188,),
                        Token(
                            doc_id="100035",
                            token='seizures',
                            start_idx=10189,
                            end_idx=10197,),
                        ...
                    ]

            fp: filepath use while saving CSV

        Returns:
            None
        """
        self.tokens_without_annotations = pd.DataFrame(tokens)
        tokens_df_schema.validate(self.tokens_without_annotations)
        self._write_csv(fp=fp, is_annotated=False)

    @staticmethod
    @check_output(annotations_df_schema)
    def read_csv_tokens_with_annotations(
        fp: Union[Path, str]
    ) -> pd.core.frame.DataFrame:
        """Read CSV file as dataframe with annotated tokens.

        Args:
            fp: filepath to CSV

        Returns:
            annotations: dataframe with annotated tokens,
                columns: [doc_id, token, start_idx, end_idx, entity]
        """
        return pd.read_csv(fp, converters=annotations_col_converters)

    @staticmethod
    @check_output(tokens_df_schema)
    def read_csv_tokens_without_annotations(
        fp: Union[Path, str]
    ) -> pd.core.frame.DataFrame:
        """Read CSV file as dataframe with unannotated tokens.

        Args:
            fp: filepath to CSV

        Returns:
            tokens: dataframe with unannotated tokens
                columns: [doc_id, token, start_idx, end_idx]
        """
        return pd.read_csv(fp, converters=tokens_col_converters)
