"""This module contains methods to build raw EHR data from file."""
from abc import ABC
from pathlib import Path
from typing import Union, List
from ner_ehr.data.entities import AnnotationTuple, TokenTuple
from ner_ehr.callbacks import AnnotationParser
from ner_ehr.tokenizers import Tokenizer
from ner_ehr.utils import copy_docstring


class Builder(ABC):
    """Base Builder class to build text from a file."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(
        str,
        fp: Union[str, Path],
    ) -> Union[List[TokenTuple], List[AnnotationTuple]]:
        pass


class AnnotationBuilder(Builder):
    """Read, parse and build annotations
    from file with NER annotations."""

    @copy_docstring(AnnotationParser.__init__)
    def __init__(self, parser: AnnotationParser):
        super().__init__(
            parser=parser,
        )

    def build(
        self,
        fp: Union[Path, str],
    ) -> List[AnnotationTuple]:
        """Build and generate annotations from file.

        Args:
            fp: file path
                Assumes file containing NER annotations.
                For ex: `Reason 10179 10188`: token with characters
                    between indexes 10179 and 10188 (inclusive) belong to `Reason` entity.

        Returns:
            A list of NamedTuple
                Ex: [
                        Annotation(token='recurrent', start_idx=10179, end_idx=10188, tag='B-Reason'),
                        Annotation(token='seizures', start_idx=10189, end_idx=10197, tag='I-Reason'),
                    ...
                    ]
        """
        annotations = []
        with open(fp, "r") as f:
            for line in f.readlines():
                line = self.parser(
                    raw_annotation=line,
                )
                annotations += line
        return annotations


class TokenBuilder(Builder):
    """Read and build tokens from file."""

    def __init__(self, tokenizer: Tokenizer):
        super().__init__(tokenizer=tokenizer)

    def build(
        self,
        fp: Union[Path, str],
    ) -> List[TokenTuple]:
        """Build and return annotation from file.

        Args:
            fp: file path

        Returns:
            A list of NamedTuple
                Ex: [
                        Token(token='Admission', start_idx=0, end_idx=9),
                        Token(token='Date', start_idx=10, end_idx=14),
                        ...
                    ]
        """
        with open(fp, "r") as f:
            text = f.read()
        return self.tokenizer(text)
