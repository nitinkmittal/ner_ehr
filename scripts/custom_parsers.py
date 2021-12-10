"""This script contains custom parsers to
    generate annotated tokens from EHR annotations and
    unannotated tokens from EHR text-file."""
import os
from pathlib import Path
from typing import List, Optional, Union

from ner_ehr.data.utils import sort_namedtuples
from ner_ehr.data.variables import (
    Annotation,
    AnnotationTuple,
    Token,
    TokenTuple,
)
from ner_ehr.parsers import AnnotationParser, TokenParser
from ner_ehr.tokenizers import Tokenizer, _validate_token_idxs
from ner_ehr.utils import copy_docstring


def read_record(
    record_fp: Union[Path, str],
) -> str:
    """Read EHR text-file from given filepath.

    Args:
        record_fp: filepath to EHR text-file

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
    """Extract filename as record_id/doc_id from given filepath.

    Args:
        record_fp: filepath to EHR text-file

        remove_ext: if True, file extension is removed
            otherwise not

    Returns:
        string record_id/doc_id
    """

    filename = os.path.basename(record_fp)
    if remove_ext:
        filename = filename.split(".")[0]
    return filename


class CustomAnnotationParser(AnnotationParser):
    """Parser for generating list of annotated tokens
    from EHR annotations."""

    @copy_docstring(Tokenizer.__init__)
    def __init__(
        self,
        tokenizer: Tokenizer,
    ):
        super().__init__(
            tokenizer=tokenizer,
        )

    def _read_annotations(self, annotations_fp: Union[Path, str]) -> List[str]:
        """Read EHR annotations from given filepath to text-file.

        Args:
            annotations_fp: filepath to text-file with annotations
                Sample annotations in text-file:
                T1  Reason  10179   10197	recurrent seizures
                R1	Reason-Drug Arg1:T1 Arg2:T3
                T3	Drug 10227 10233	ativan
                ...

        Returns:
            A list of string annotations
                [
                    T1  Reason  10179   10197	recurrent seizures,
                    T3	Drug 10227 10233	ativan,
                    ...,
                ]
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
        doc_id: Optional[str] = None,
    ) -> List[AnnotationTuple]:
        """Parse annotations to generate list of annotated tokens.

        Args:
            annotations_fp: filepath to EHR annotations, ex: 100035.ann
                Sample annotations in text-file:
                    T1  Reason  10179   10197	recurrent seizures
                    R1	Reason-Drug Arg1:T1 Arg2:T3
                    T3	Drug 10227 10233	ativan
                    ...

            record_fp: filepath to EHR text-file

            doc_id: unique identifier for document/sample from which
                tokens/annotations are generated, default=None
                if None, then `doc_id` inferred from record_fp

        Returns:
            A list of annotated tuples
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
                        ...,
                    ]
        """

        self.annotations: List[
            AnnotationTuple
        ] = []  # reinitialize list of AnnotationTuples

        # muting start and end character index checking while generating tokens
        validate_token_idxs: bool = self.tokenizer.validate_token_idxs
        self.tokenizer.validate_token_idxs = False

        # reading EHR annotations
        annotations = self._read_annotations(annotations_fp=annotations_fp)
        # reading EHR text-file
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

        # validating start and end character indexing if `validate_token_idxs`
        #   is True
        if validate_token_idxs:
            _validate_token_idxs(tokens=self.annotations, text=record)

        # resetting flag `validate_token_idxs` to it's original value
        self.tokenizer.validate_token_idxs = validate_token_idxs

        # sorting annotated tokens
        return sort_namedtuples(
            self.annotations, by=["doc_id", "start_idx"], ascending=True
        )


class CustomTokenParser(TokenParser):
    """Parser for generating list of unannotated tokens
    from EHR text."""

    @copy_docstring(Tokenizer.__init__)
    def __init__(
        self,
        tokenizer: Tokenizer,
    ):
        super().__init__(
            tokenizer=tokenizer,
        )

    def _parse(self, doc_id: str, substring: str, shift: int) -> None:
        """Parse substrings within EHR text.

        Args:
            doc_id: unique identifier for document/sample from which
                token is generated

            substring: substring from EHR text

            shift: integer value to shift start and end character indexes
                for each token
        """
        for token in self.tokenizer(doc_id=doc_id, text=substring):
            token = token._replace(
                start_idx=token.start_idx + shift,
                end_idx=token.end_idx + shift,
            )
            self.tokens.append(token)

    def parse(
        self,
        record_fp: Union[Path, str],
        annotations: List[AnnotationTuple] = [],
        doc_id: Optional[str] = None,
    ) -> List[TokenTuple]:
        """Parse EHR text to generate list of unannotated token tuples.

        Note: this parser requires list of annotated token tuples
            before parsing EHR text if annotations are already available
            for given EHR text. Annotated token tuples are required
            to preserve all original annotations and avoid incorrect
            tokenization of unstructured EHR text.

        Args:
            record_fp: filepath to EHR text-file

            annotations: a list of AnnotationTuples, default=[] (empty list)
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
                        ...,
                    ]

            doc_id: unique identifier for document/sample from which
                tokens/annotations are generated, default=None
                if None, then `doc_id` inferred from record_fp

        Returns:
            A list of unannotated token tuples
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
                    ...,
                    ]
        """
        self.tokens: List[TokenTuple] = []  # reinitialize list of TokenTuples

        # muting start and end character index checking while generating tokens
        validate_token_idxs: bool = self.tokenizer.validate_token_idxs
        self.tokenizer.validate_token_idxs = False

        record: str = read_record(record_fp=record_fp)
        start_idx: int = 0
        end_idx: int = 0

        if doc_id is None:
            doc_id = extract_record_id(record_fp=record_fp)

        for i, ann in enumerate(annotations):
            end_idx = ann.start_idx
            # adding tokens before this annotation
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

        # validating start and end character indexing if `validate_token_idxs`
        #   is True
        if validate_token_idxs:
            _validate_token_idxs(tokens=self.tokens, text=record)

        # resetting flag `validate_token_idxs` to it's original value
        self.tokenizer.validate_token_idxs = validate_token_idxs

        # sorting unannotated tokens
        return sort_namedtuples(
            self.tokens, by=["doc_id", "start_idx"], ascending=True
        )
