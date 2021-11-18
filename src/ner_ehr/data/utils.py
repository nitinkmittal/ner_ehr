from ner_ehr.data.variables import TokenTuple, AnnotationTuple
from typing import List, Union
import pandas as pd


def sort_namedtuples(
    namedtuples: Union[List[TokenTuple], List[AnnotationTuple]],
):
    df = pd.DataFrame(namedtuples).sort_values("start_idx")
    return list(df.itertuples(name=type(namedtuples[0]).__name__, index=False))
