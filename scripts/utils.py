import os
from glob import glob
from pathlib import Path
from typing import List, Union

from ner_ehr.data.ehr import EHR
from ner_ehr.data.utils import df_to_namedtuples
from ner_ehr.data.variables import AnnotationTuple


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
