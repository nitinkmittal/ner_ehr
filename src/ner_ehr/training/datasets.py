"""This module contain PyTorch Dataset and DataLoader,
    Pytorch Lightning DataModules."""
import os
from abc import ABC
from glob import glob
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import torch
from pytorch_lightning import LightningDataModule
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from ner_ehr.data import Constants
from ner_ehr.data.ehr import EHR
from ner_ehr.data.utils import (
    df_to_namedtuples,
    generate_token_seqs,
)
from ner_ehr.data.variables import (
    AnnotationTuple,
    LongAnnotationTuple,
    TokenTuple,
)

from ner_ehr.data.vocab import TokenEntityVocab

DEFAULT_ANNOTATED: bool = False
DEFAULT_SEQ_LENGTH: int = 256
DEFAULT_BATCH_SIZE: int = 1
DEFAULT_SHUFFLE_TRAIN: bool = True
DEFAULT_SHUFFLE_VAL: bool = True
DEFAULT_SHUFFLE_TEST: bool = False
DEFAULT_NUM_WORKERS: int = 1


class EHRDataset(Dataset):
    def __init__(
        self,
        dir: Union[Path, str],
        vocab: TokenEntityVocab,
        annotated: bool = DEFAULT_ANNOTATED,
        seq_length: int = DEFAULT_SEQ_LENGTH,
    ):
        """
        Args:
            dir: directory containing CSVs with annotated tokens

            vocab: ner_data.utils.TokenEntityVocab object trained on annotationtuples

            annotated: boolean flag
                if True, tokens with annotations are read,
                otherwise without annotations are read

            seq_length: maximum number of tokens in a sequence
                default: 256
        """
        self.dir = dir
        self.vocab = vocab
        self.annotated = annotated
        self.seq_length = seq_length
        self.seqs: List[List[AnnotationTuple]] = []
        self._setup()

    def _setup(
        self,
    ):
        """Helper function to read and generate sequences of annotated tokens.

        Note: if `annotated` flag is False, then outside entity label is
            added to all tokens
        """
        # reading CSVs one-by-one and generate sequences of tokens
        for fp in glob(os.path.join(self.dir, r"*.csv")):
            if self.annotated:
                annotatedtuples = df_to_namedtuples(
                    name=AnnotationTuple.__name__,
                    df=EHR.read_csv_tokens_with_annotations(fp=fp),
                )
            else:
                annotatedtuples = EHR.read_csv_tokens_without_annotations(
                    fp=fp
                )
                # adding `OUTSIDE`/`UNTAG` entity label by default
                annotatedtuples["entity"] = Constants.UNTAG_ENTITY_LABEL.value
                annotatedtuples = df_to_namedtuples(
                    name=AnnotationTuple.__name__,
                    df=annotatedtuples,
                )

            # converting annotatedtuples to long_annotatedtuples
            #   i.e adding token indexes and entity labels from pre-trained vocab
            annotatedtuples = [
                self.vocab.annotation_to_longannotation(
                    annotatedtuple=annotatedtuple
                )
                for annotatedtuple in annotatedtuples
            ]
            self.seqs += generate_token_seqs(
                annotatedtuples=annotatedtuples, seq_length=self.seq_length
            )

    def __getitem__(
        self, i: int
    ) -> Union[List[TokenTuple], List[AnnotationTuple]]:
        return self.seqs[i]

    def __len__(
        self,
    ) -> int:
        return len(self.seqs)


class EHRBatchCollator(ABC):
    """Helper function to prepare batch for RNNs"""

    def __init__(self, return_meta: bool = False):
        """
        Args:
            return_meta: a boolean flag
                if True, metadata about current batch is also returned
                Note: metadata is return in form of list of tuples,
                    because of PyTorch collate_fn requirement
        """
        self.return_meta = return_meta

    def __call__(
        self, batch: List[LongAnnotationTuple]
    ) -> Tuple[
        Tuple[torch.Tensor, torch.Tensor],
        Optional[List[Tuple[str, str, int, int, str]]],
    ]:
        # Note: pad_sequences require tuple/list of tensors here
        #   with batch_first=True
        X = pad_sequence(
            [
                torch.tensor(
                    [
                        long_annotatedtuple.token_idx
                        for long_annotatedtuple in long_annotatedtuples
                    ],
                    dtype=torch.long,
                )
                for long_annotatedtuples in batch
            ],
            batch_first=True,
            padding_value=Constants.PAD_TOKEN_IDX.value,
        )
        Y = pad_sequence(
            [
                torch.tensor(
                    [
                        long_annotatedtuple.entity_label
                        for long_annotatedtuple in long_annotatedtuples
                    ],
                    dtype=torch.long,
                )
                for long_annotatedtuples in batch
            ],
            batch_first=True,
            padding_value=Constants.UNTAG_ENTITY_INT_LABEL.value,
        )

        if self.return_meta:
            meta = [
                [
                    [
                        long_annotatedtuple.doc_id,
                        long_annotatedtuple.token,
                        long_annotatedtuple.start_idx,
                        long_annotatedtuple.end_idx,
                        long_annotatedtuple.entity,
                    ]
                    for long_annotatedtuple in long_annotatedtuples
                ]
                for long_annotatedtuples in batch
            ]
            return X, Y, meta

        return X, Y, None


DEFAULT_COLLATE_FUNC = EHRBatchCollator(return_meta=True)


class EHRDataModule(LightningDataModule):
    """PyTorch Lightning module for EHRDataset."""

    def __init__(
        self,
        vocab: TokenEntityVocab,
        seq_length: int = DEFAULT_SEQ_LENGTH,
        annotated: bool = DEFAULT_ANNOTATED,
        dir_train: Optional[Union[Path, str]] = None,
        dir_val: Optional[Union[Path, str]] = None,
        dir_test: Optional[Union[Path, str]] = None,
        batch_size_train: int = DEFAULT_BATCH_SIZE,
        batch_size_val: int = DEFAULT_BATCH_SIZE,
        batch_size_test: int = DEFAULT_BATCH_SIZE,
        num_workers_train: int = DEFAULT_NUM_WORKERS,
        num_workers_val: int = DEFAULT_NUM_WORKERS,
        num_workers_test: int = DEFAULT_NUM_WORKERS,
        shuffle_train: bool = DEFAULT_SHUFFLE_TRAIN,
        shuffle_val: bool = DEFAULT_SHUFFLE_VAL,
        shuffle_test: bool = DEFAULT_SHUFFLE_TEST,
        collate_fn_train: Callable[
            [List[LongAnnotationTuple]],
            Tuple[
                Tuple[torch.Tensor, torch.Tensor],
                Optional[List[Tuple[str, str, int, int, str]]],
            ],
        ] = DEFAULT_COLLATE_FUNC,
        collate_fn_val: Callable[
            [List[LongAnnotationTuple]],
            Tuple[
                Tuple[torch.Tensor, torch.Tensor],
                Optional[List[Tuple[str, str, int, int, str]]],
            ],
        ] = DEFAULT_COLLATE_FUNC,
        collate_fn_test: Callable[
            [List[LongAnnotationTuple]],
            Tuple[
                Tuple[torch.Tensor, torch.Tensor],
                Optional[List[Tuple[str, str, int, int, str]]],
            ],
        ] = DEFAULT_COLLATE_FUNC,
    ):
        super().__init__()
        self.vocab = vocab
        self.seq_length = seq_length
        self.annotated = annotated
        self.dir_train = dir_train
        self.dir_val = dir_val
        self.dir_test = dir_test
        self.train_dataset: EHRDataset = None
        self.val_dataset: EHRDataset = None
        self.test_dataset: EHRDataset = None
        self.batch_size_train = batch_size_train
        self.batch_size_val = batch_size_val
        self.batch_size_test = batch_size_test
        self.num_workers_train = num_workers_train
        self.num_workers_val = num_workers_val
        self.num_workers_test = num_workers_test
        self.shuffle_train = shuffle_train
        self.shuffle_val = shuffle_val
        self.shuffle_test = shuffle_test
        self.collate_fn_train = collate_fn_train
        self.collate_fn_val = collate_fn_val
        self.collate_fn_test = collate_fn_test

    def setup(self, stage: Optional[str] = None):
        if self.dir_train is not None:
            self.train_dataset = EHRDataset(
                dir=self.dir_train,
                vocab=self.vocab,
                annotated=self.annotated,
                seq_length=self.seq_length,
            )
        if self.dir_val is not None:
            self.val_dataset = EHRDataset(
                dir=self.dir_val,
                vocab=self.vocab,
                annotated=self.annotated,
                seq_length=self.seq_length,
            )
        if self.dir_test is not None:
            self.test_dataset = EHRDataset(
                dir=self.dir_test,
                vocab=self.vocab,
                annotated=self.annotated,
                seq_length=self.seq_length,
            )

    def train_dataloader(self):
        if self.train_dataset is None:
            return
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size_train,
            shuffle=self.shuffle_train,
            num_workers=self.num_workers_train,
            collate_fn=self.collate_fn_train,
        )

    def val_dataloader(self):
        if self.val_dataset is None:
            return
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size_val,
            shuffle=self.shuffle_val,
            num_workers=self.num_workers_val,
            collate_fn=self.collate_fn_val,
        )

    def test_dataloader(self):
        if self.test_dataset is None:
            return
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size_test,
            shuffle=self.shuffle_test,
            num_workers=self.num_workers_test,
            collate_fn=self.collate_fn_test,
        )
