from abc import ABC

from typing import Union, List
from ner_ehr.data.variables import AnnotationTuple, TokenTuple
import pandas as pd
from pathlib import Path
import os

UNTAG_ENTITY_LABEL = "O"


class CoNLLDataset(ABC):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

        self.tokens_with_annotations: pd.core.frame.DataFrame = None
        self.tokens_without_annotations: pd.core.frame.DataFrame = None

    def _to_csv(
        self, fp: Union[str, Path], is_annotated: bool
    ) -> Union[str, Path]:
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

    def to_csv_tokens_with_annotations(
        self,
        tokens: List[TokenTuple],
        annotations: List[AnnotationTuple],
        fp: Union[str, Path],
    ) -> None:

        self.tokens_with_annotations = pd.merge(
            pd.DataFrame(tokens),
            pd.DataFrame(annotations),
            on=tokens[0]._fields,
            how="left",
        )

        self.tokens_with_annotations["tag"].fillna(
            UNTAG_ENTITY_LABEL, inplace=True
        )
        self._to_csv(fp=fp, is_annotated=True)

    def to_csv_tokens_without_annotations(
        self,
        tokens: List[TokenTuple],
        fp: Union[str, Path],
    ) -> None:

        self.tokens_without_annotations = pd.DataFrame(tokens)
        self._to_csv(fp=fp, is_annotated=False)


class EHR(CoNLLDataset):
    def __init__(
        self,
    ):
        super().__init__()
