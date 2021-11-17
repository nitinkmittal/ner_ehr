"""Custom methods to process annotations."""
from typing import List

from ner_ehr.data.entities import AnnotationTuple, Annotation
from ner_ehr.callbacks import AnnotationParser

from ner_ehr.tokenizers import SplitTokenizer, Tokenizer, ScispacyTokenizer


from functools import partial


def parser(
    raw_annotation: str,
    raw_text: str,
    tokenizer: Tokenizer,
) -> List[AnnotationTuple]:
    """To parse given string."""

    annotations = []
    raw_annotation = raw_annotation.strip()  # removing whitespace

    # if empty line or not an annotation
    if not raw_annotation or raw_annotation[0] != "T":
        return annotations

    # T150---->Frequency 15066 15076;15077 15095---->Q6H (every 6 hours) as needed
    _, annotation, _ = raw_annotation.split("\t")  # tab separated values

    annotation = annotation.split(";")

    tag, start_idx, end_idx = annotation.pop(0).split()

    while annotation:
        _, end_idx = annotation.pop(0).split()

    start_idx = int(start_idx)
    end_idx = int(end_idx)

    tokens = tokenizer(text=raw_text[start_idx:end_idx])
    for i, token in enumerate(tokens):
        if i == 0:
            _tag = f"B-{tag}"
        else:
            _tag = f"I-{tag}"

        annotation = Annotation(
            token=token.token,
            start_idx=start_idx + token.start_idx,
            end_idx=start_idx + token.start_idx + len(token.token),
            tag=_tag,
        )
        annotations.append(annotation())

    return annotations
