from typing import Union, List
from pathlib import Path
from ner_ehr.data.variables import AnnotationTuple, LongAnnotationTuple
from ner_ehr.data.utils import df_to_namedtuples
from ner_ehr.data.ehr import EHR
import os
from glob import glob
from sklearn.preprocessing import LabelEncoder

from ner_ehr.data.vocab import TokenEntityVocab


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


def build_vocab(annotatedtuples: List[AnnotationTuple]) -> TokenEntityVocab:
    """Build vocab from list of annotated tuples.

    Args:
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

    Returns:
        TokenEntityVocab object
    """
    vocab = TokenEntityVocab()
    vocab.fit(annotatedtuples=annotatedtuples)
    return vocab


def label_encoder(
    annotatedtuples: Union[List[AnnotationTuple], List[LongAnnotationTuple]]
) -> LabelEncoder:
    encoder = LabelEncoder()
    encoder.fit([annotatedtuple.entity for annotatedtuple in annotatedtuples])
    return encoder
