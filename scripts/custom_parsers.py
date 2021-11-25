"""Custom methods to generate annotated tokens."""
from pathlib import Path
from typing import List, Union, Optional
import os
from ner_ehr.data.utils import sort_namedtuples
from ner_ehr.data.variables import (
    Annotation,
    AnnotationTuple,
    Token,
    TokenTuple,
)
from ner_ehr.parsers import AnnotationParser, TokenParser
from ner_ehr.tokenizers import Tokenizer, _validate_token_idxs


def read_record(
    record_fp: Union[Path, str],
) -> str:
    """Read EHR from given filepath.

    Args:
        record_fp: filepath to EHR

    Returns:
        String text from EHR
    """
    with open(record_fp, "r") as f:
        record = f.read()

    # replacing double quotes necessary to avoid
    #   incorrect character indexing for tokens
    record = str(record).replace('"', "'")
    return record


def extract_record_id(
    record_fp: Union[Path, str], remove_ext: bool = True
) -> str:
    """Extract filename from given filepath.

    Args:
        record_fp: filepath to EHR, ex: 100035.txt

        remove_ext: if True, file extension is removed
            otherwise not

    Returns:
        record filename
    """

    filename = os.path.basename(record_fp)
    if remove_ext:
        filename = filename.split(".")[0]

    return filename


class CustomAnnotationParser(AnnotationParser):
    """Parser to process an annotation."""

    def __init__(
        self,
        tokenizer: Tokenizer,
    ):
        """
        Args:
            tokenizer (callable): to convert string into list of tokens
            >>> tokenizer.tokenize('this is a sentence') ->
                ['this', 'is', 'a', 'sentence']
        """

        super().__init__(
            tokenizer=tokenizer,
        )

    def _read_annotations(self, annotations_fp: Union[Path, str]) -> List[str]:
        """Read annotations from given filepath.

        Args:
            annotations_fp: filepath to annotations

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

    @sort_namedtuples
    def parse(
        self,
        annotations_fp: Union[Path, str],
        record_fp: Union[Path, str],
        doc_id: Optional[str] = None,
    ) -> List[AnnotationTuple]:
        """Parse annotations.

        Args:
            annotations_fp: filepath to annotations, ex: 100035.ann
                Assumes file containing NER annotations.
                For ex: `Reason 10179 10188`: token with characters
                    between indexes 10179 and 10188 (inclusive)
                    belong to `Reason` entity.

            record_fp: filepath to EHR, ex: 100035.txt

            doc_id: unique identifier for document/sample from which
                variable is generated. Default: None
                if None, then doc_id inferred from record_fp

        Returns:
            A list of AnnotationTuples
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
        """

        self.annotations: List[
            AnnotationTuple
        ] = []  # reinitialize list of AnnotationTuples
        validate_token_idxs: bool = self.tokenizer.validate_token_idxs

        # turning off this flag to avoid character index validation for substrings
        self.tokenizer.validate_token_idxs = False

        annotations = self._read_annotations(annotations_fp=annotations_fp)
        record = read_record(record_fp=record_fp)

        if doc_id is None:
            doc_id = extract_record_id(record_fp=record_fp)

        for annotation in annotations:
            # T150\tFrequency 15066 15076;15077 15095\tQ6H (every 6 hours) as needed
            #   -> Frequency 15066 15076;15077 15095
            _, annotation, _ = annotation.split("\t")  # tab separated values

            # Frequency 15066 15076;15077 15095 -> [Frequency 15066 15076, 15077 15095]
            annotation = annotation.split(";")

            # Frequency 15066 15076 -> [Frequency, 15066, 15076]
            entity, start_idx, end_idx = annotation.pop(0).split()
            while (
                annotation
            ):  # fetching last character index for an annotation
                _, end_idx = annotation.pop(0).split()

            start_idx = int(start_idx)
            end_idx = int(end_idx)
            tokens = self.tokenizer(
                doc_id=doc_id, text=record[start_idx:end_idx]
            )
            for i, token in enumerate(tokens):
                if i == 0:
                    _entity = f"B-{entity}"
                else:
                    _entity = f"I-{entity}"

                annotation = Annotation(
                    doc_id=token.doc_id,
                    token=token.token,
                    start_idx=start_idx + token.start_idx,
                    end_idx=start_idx + token.start_idx + len(token.token),
                    entity=_entity,
                )()
                self.annotations.append(annotation)

        if validate_token_idxs:
            _validate_token_idxs(tokens=self.annotations, text=record)

        # resetting this flag to avoid character index validation for substrings
        self.tokenizer.validate_token_idxs = validate_token_idxs

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

    def _parse(self, doc_id: str, substring: str, shift: int):
        """Parse substrings within EHR.

        Args:
            doc_id: unique identifier for document/sample from which
                variable is generated

            substring: substring from EHR

            shift: to shift start and end character indexes of given token
        """
        for token in self.tokenizer(doc_id=doc_id, text=substring):
            token = token._replace(
                start_idx=token.start_idx + shift,
                end_idx=token.end_idx + shift,
            )
            self.tokens.append(token)

    @sort_namedtuples
    def parse(
        self,
        record_fp: Union[Path, str],
        annotations: List[AnnotationTuple],
        doc_id: Optional[str] = None,
    ) -> List[TokenTuple]:
        """Parse EHR.

        Note: This parser requires list of AnnotationTuples
            to preserve all annotated tokens while generating new tokens from EHR.

        Args:
            record_fp: filepath to EHR, ex: 100035.txt

            annotations: a list of AnnotationTuples
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

            doc_id: unique identifier for document/sample from which
                variable is generated. Default: None
                if None, then doc_id inferred from record_fp

        Returns:
            A list of TokenTuples
                Ex: [
                        TokenTuple(
                            doc_id="100035",
                            token='recurrent',
                            start_idx=10179,
                            end_idx=10188),
                        TokenTuple(
                            doc_id="100035",
                            token='seizures',
                            start_idx=10189,
                            end_idx=10197),
                    ...
                    ]
        """
        self.tokens: List[TokenTuple] = []  # reinitialize list of TokenTuples
        # muting index checking while generating tokens
        validate_token_idxs: bool = self.tokenizer.validate_token_idxs

        # turning off this flag to avoid character index validation for substrings
        self.tokenizer.validate_token_idxs = False

        record: str = read_record(record_fp=record_fp)
        start_idx: int = 0
        end_idx: int = 0

        if doc_id is None:
            doc_id = extract_record_id(record_fp=record_fp)

        for i, ann in enumerate(annotations):
            end_idx = ann.start_idx
            # adding tokens before annotation
            self._parse(
                doc_id=doc_id,
                substring=record[start_idx:end_idx],
                shift=start_idx,
            )
            # adding token from annotation
            self.tokens.append(
                Token(
                    doc_id=doc_id,
                    token=ann.token,
                    start_idx=ann.start_idx,
                    end_idx=ann.end_idx,
                )()
            )
            start_idx = ann.end_idx

        # addding tokens after all annotations
        self._parse(
            doc_id=doc_id, substring=record[start_idx:], shift=start_idx
        )

        if validate_token_idxs:
            _validate_token_idxs(tokens=self.tokens, text=record)

        # resetting this flag to avoid character index validation for substrings
        self.tokenizer.validate_token_idxs = validate_token_idxs

        return self.tokens
