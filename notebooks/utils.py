"""Custom methods to process annotations."""
from typing import List

from ner_ehr.data.variables import (
    AnnotationTuple,
    Annotation,
    TokenTuple,
    Token,
)
from ner_ehr.tokenizers import Tokenizer
from ner_ehr.parsers import AnnotationParser, TokenParser
from pathlib import Path
from typing import Union
from ner_ehr.tokenizers import _validate_token_idxs
from ner_ehr.data.utils import sort_namedtuples


def read_record(
    record_fp: Union[Path, str],
) -> str:
    """Read EHR from given file path.

    Args:
        record_fp: file path to EHR

    Returns:
        String text from EHR
    """
    with open(record_fp, "r") as f:
        record = f.read()
    return record


class CustomAnnotationParser(AnnotationParser):
    """Parser to process an annotation."""

    def __init__(
        self,
        tokenizer: Tokenizer,
    ):
        """
        Args:
            tokenizer (callable): to convert string into list of tokens
            >>> tokenizer.tokenize('this is a sentence') -> ['this', 'is', 'a', 'sentence']
        """

        super().__init__(
            tokenizer=tokenizer,
        )

    def _read_annotations(self, annotations_fp: Union[Path, str]) -> List[str]:
        """Read annotations from given file path.

        Args:
            annotations_fp: file path to annotations

        Returns:
            A list of string annotations
        """
        annotations = []
        with open(annotations_fp, "r") as f:
            for annotation in f.readlines():
                annotation = str(annotation).strip()  # removing whitespace

                # if empty or not an NER annotation
                if not annotation or annotation[0] != "T":
                    continue
                annotations.append(annotation)
        return annotations

    def parse(
        self,
        annotations_fp: Union[Path, str],
        record_fp: Union[Path, str],
    ) -> List[AnnotationTuple]:
        """Parse annotations.

        Args:
            annotations_fp: file path to annotations
                Assumes file containing NER annotations.
                For ex: `Reason 10179 10188`: token with characters
                    between indexes 10179 and 10188 (inclusive) belong to `Reason` entity.

            record_fp: file path to EHR

        Returns:
            A list of NamedTuple
                Ex: [
                        Annotation(token='recurrent', start_idx=10179, end_idx=10188, tag='B-Reason'),
                        Annotation(token='seizures', start_idx=10189, end_idx=10197, tag='I-Reason'),
                    ...
                    ]
        """

        annotations = self._read_annotations(annotations_fp=annotations_fp)
        record = read_record(record_fp=record_fp)
        validate_token_idxs: bool = self.tokenizer.validate_token_idxs
        self.tokenizer.validate_token_idxs = False

        for annotation in annotations:
            # T150---->Frequency 15066 15076;15077 15095---->Q6H (every 6 hours) as needed
            _, annotation, _ = annotation.split("\t")  # tab separated values
            annotation = annotation.split(";")
            tag, start_idx, end_idx = annotation.pop(0).split()
            while annotation:  # fetching last character index for annotation
                _, end_idx = annotation.pop(0).split()
            start_idx = int(start_idx)
            end_idx = int(end_idx)
            tokens = self.tokenizer(text=record[start_idx:end_idx])
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
                )()
                self.annotations.append(annotation)

        if validate_token_idxs:
            _validate_token_idxs(tokens=self.annotations, text=record)

        # reset tokenizer
        self.tokenizer.validate_token_idxs = validate_token_idxs
        self.annotations = sort_namedtuples(namedtuples=self.annotations)
        return self.annotations


class CustomTokenParser(TokenParser):
    """Parser to process an annotation."""

    def __init__(
        self,
        tokenizer: Tokenizer,
    ):
        """
        Args:
        tokenizer (callable): to convert string into list of tokens
            >>> tokenizer.tokenize('this is a sentence') -> ['this', 'is', 'a', 'sentence']
        """
        super().__init__(
            tokenizer=tokenizer,
        )

    def parse(
        self, record_fp: Union[Path, str], annotations: List[AnnotationTuple]
    ) -> List[TokenTuple]:
        """Parse EHR.

        Args:
            annotations_fp: file path to annotations
                Assumes file containing NER annotations.
                For ex: `Reason 10179 10188`: token with characters
                    between indexes 10179 and 10188 (inclusive) belong to `Reason` entity.

            record_fp: file path to EHR

        Returns:
            A list of NamedTuple
                Ex: [
                        Annotation(token='recurrent', start_idx=10179, end_idx=10188, tag='B-Reason'),
                        Annotation(token='seizures', start_idx=10189, end_idx=10197, tag='I-Reason'),
                    ...
                    ]
        """

        record = read_record(record_fp=record_fp)

        start = 0
        end = 0
        for i, ann in enumerate(annotations):
            end = ann.start_idx
            for token in self.tokenizer(record[start:end]):
                token = token._replace(
                    start_idx=token.start_idx + start,
                    end_idx=token.end_idx + start,
                )
                self.tokens.append(token)
            self.tokens.append(
                Token(
                    token=ann.token,
                    start_idx=ann.start_idx,
                    end_idx=ann.end_idx,
                )()
            )
            start = ann.end_idx

        self.tokens += self.tokenizer(record[start:])

        return self.tokens
