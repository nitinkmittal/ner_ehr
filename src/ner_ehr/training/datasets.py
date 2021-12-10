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
from ner_ehr.data.utils import df_to_namedtuples, generate_token_seqs
from ner_ehr.data.variables import AnnotationTuple, LongAnnotationTuple
from ner_ehr.data.vocab import TokenEntityVocab

SEQ_LENGTH: int = 256
ANNOTATED: bool = False
BATCH_SIZE: int = 1
SHUFFLE_TRAIN: bool = True
SHUFFLE_VAL: bool = False
SHUFFLE_TEST: bool = False
NUM_WORKERS: int = 0
RETURN_META: bool = False


class EHRDataset(Dataset):
    """PyTorch Dataset version to load EHR data."""

    def __init__(
        self,
        dir: Union[Path, str],
        vocab: TokenEntityVocab,
        annotated: bool = ANNOTATED,
        seq_length: int = SEQ_LENGTH,
    ):
        """
        Args:
            dir: directory containing CSVs with annotated/unannotated tokens
                if annotated, CSVs should have following columns:
                    [`doc_id`, `token`, `start_idx`, `end_idx`, `entity`]
                if unannotated, CSVs should have following columns:
                    [`doc_id`, `token`, `start_idx`, `end_idx`]

            vocab: pre-trained ner_data.vocab.TokenEntityVocab object

            seq_length: maximum number of tokens in a sequence,
                default=256

            annotated: boolean flag, default=False
                if True, tokens with annotations are read,
                otherwise without annotations are read
        """
        self.dir = dir
        self.vocab = vocab
        self.annotated = annotated
        self.seq_length = seq_length
        self.seqs: List[List[LongAnnotationTuple]] = []
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
            # i.e adding token indexes and entity labels from pre-trained vocab
            annotatedtuples = [
                self.vocab.annotation_to_longannotation(
                    annotatedtuple=annotatedtuple
                )
                for annotatedtuple in annotatedtuples
            ]
            self.seqs += generate_token_seqs(
                annotatedtuples=annotatedtuples, seq_length=self.seq_length
            )

    def __getitem__(self, i: int) -> List[LongAnnotationTuple]:
        return self.seqs[i]

    def __len__(
        self,
    ) -> int:
        return len(self.seqs)


class EHRBatchCollator(ABC):
    """Helper function to prepare batch for RNNs"""

    def __init__(self, return_meta: bool = RETURN_META):
        """
        Args:
            return_meta: a boolean flag
                if True, metadata about current batch is also returned
                Note: metadata is return in form of list of tuples,
                    because of PyTorch collate_fn requirement
        """
        self.return_meta = return_meta

    def __call__(
        self, batch: List[List[LongAnnotationTuple]]
    ) -> Tuple[
        Tuple[torch.LongTensor, torch.LongTensor],
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
            padding_value=Constants.PAD_TOKEN_ENTITY_INT_LABEL.value,
        )

        # Note: meta is not padded, not seq-length can be identified from here
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


COLLATE_FUNC = EHRBatchCollator(return_meta=RETURN_META)


class EHRDataModule(LightningDataModule):
    """PyTorch Lightning module for EHR dataset."""

    def __init__(
        self,
        vocab: TokenEntityVocab,
        seq_length: int = SEQ_LENGTH,
        annotated: bool = ANNOTATED,
        dir_train: Optional[Union[Path, str]] = None,
        dir_val: Optional[Union[Path, str]] = None,
        dir_test: Optional[Union[Path, str]] = None,
        batch_size_train: int = BATCH_SIZE,
        batch_size_val: int = BATCH_SIZE,
        batch_size_test: int = BATCH_SIZE,
        num_workers_train: int = NUM_WORKERS,
        num_workers_val: int = NUM_WORKERS,
        num_workers_test: int = NUM_WORKERS,
        shuffle_train: bool = SHUFFLE_TRAIN,
        shuffle_val: bool = SHUFFLE_VAL,
        shuffle_test: bool = SHUFFLE_TEST,
        collate_fn_train: Callable[
            [List[LongAnnotationTuple]],
            Tuple[
                Tuple[torch.Tensor, torch.Tensor],
                Optional[List[Tuple[str, str, int, int, str]]],
            ],
        ] = COLLATE_FUNC,
        collate_fn_val: Callable[
            [List[LongAnnotationTuple]],
            Tuple[
                Tuple[torch.Tensor, torch.Tensor],
                Optional[List[Tuple[str, str, int, int, str]]],
            ],
        ] = COLLATE_FUNC,
        collate_fn_test: Callable[
            [List[LongAnnotationTuple]],
            Tuple[
                Tuple[torch.Tensor, torch.Tensor],
                Optional[List[Tuple[str, str, int, int, str]]],
            ],
        ] = COLLATE_FUNC,
    ):
        """PyTorch Lightning Datamodule for EHR dataset.

        Args:
            vocab: pre-trained ner_data.vocab.TokenEntityVocab object

            seq_length: maximum number of tokens in a sequence,
                default=256

            annotated: boolean flag, default=False
                if True, tokens with annotations are read,
                otherwise without annotations are read

            dir_train: directory containing CSVs with training
                annotated/unannotated tokens,
                if annotated, CSVs should have following columns:
                    [`doc_id`, `token`, `start_idx`, `end_idx`, `entity`]
                if unannotated, CSVs should have following columns:
                    [`doc_id`, `token`, `start_idx`, `end_idx`]
                Usually annotated tokens are provided for training

            dir_val: directory containing CSVs with validation
                annotated/unannotated tokens,
                if annotated, CSVs should have following columns:
                    [`doc_id`, `token`, `start_idx`, `end_idx`, `entity`]
                if unannotated, CSVs should have following columns:
                    [`doc_id`, `token`, `start_idx`, `end_idx`]
                Usually annotated tokens are provided for validation

            dir_test: directory containing CSVs with testing
                annotated/unannotated tokens,
                if annotated, CSVs should have following columns:
                    [`doc_id`, `token`, `start_idx`, `end_idx`, `entity`]
                if unannotated, CSVs should have following columns:
                    [`doc_id`, `token`, `start_idx`, `end_idx`]
                Usually unannotated tokens are provided for testing

            batch_size_train: number of sequences of annotated/unannotated
                tokens in a training batch, default=1

            batch_size_val: number of sequences of annotated/unannotated
                tokens in a validation batch, default=1

            batch_size_test: number of sequences of annotated/unannotated
                tokens in a testing batch, default=1

            num_workers_train: how many subprocesses to use for train
                data loading. 0 means that the data will be loaded in
                the main process, default=0

            num_workers_val: how many subprocesses to use for validation
                data loading. 0 means that the data will be loaded in
                the main process, default=0

            num_workers_test: how many subprocesses to use for test
                data loading. 0 means that the data will be loaded in
                the main process, default=0

            shuffle_train: boolean flag, default=True
                set to True to have the train data reshuffled at every epoch

            shuffle_val: boolean flag, default=False
                set to True to have the validation data reshuffled at every
                epoch

            shuffle_test: boolean flag, default=False
                set to True to have the test data reshuffled at every epoch

            collate_fn_train: merges a list of training samples to form
                a mini-batch of Tensor(s). Used when using batched loading
                from a map-style dataset

            collate_fn_val: merges a list of validation samples to form
                a mini-batch of Tensor(s). Used when using batched loading
                from a map-style dataset

            collate_fn_test: merges a list of testing samples to form
                a mini-batch of Tensor(s). Used when using batched loading
                from a map-style dataset
        """
        super().__init__()
        self.vocab = vocab
        self.seq_length = seq_length
        self.annotated = annotated
        self.dir_train = dir_train
        self.dir_val = dir_val
        self.dir_test = dir_test
        self.ds_train: EHRDataset = None
        self.ds_val: EHRDataset = None
        self.ds_train: EHRDataset = None
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
            self.ds_train = EHRDataset(
                dir=self.dir_train,
                vocab=self.vocab,
                annotated=self.annotated,
                seq_length=self.seq_length,
            )
        if self.dir_val is not None:
            self.ds_val = EHRDataset(
                dir=self.dir_val,
                vocab=self.vocab,
                annotated=self.annotated,
                seq_length=self.seq_length,
            )
        if self.dir_test is not None:
            self.ds_train = EHRDataset(
                dir=self.dir_test,
                vocab=self.vocab,
                annotated=self.annotated,
                seq_length=self.seq_length,
            )

    def train_dataloader(self):
        if self.ds_train is None:
            raise ValueError(
                "'dir_train' not provided, train dataset not initialized."
            )
        return DataLoader(
            self.ds_train,
            batch_size=self.batch_size_train,
            shuffle=self.shuffle_train,
            num_workers=self.num_workers_train,
            collate_fn=self.collate_fn_train,
        )

    def val_dataloader(self):
        if self.ds_val is None:
            raise ValueError(
                "'dir_val' not provided, validation dataset not initialized."
            )
        return DataLoader(
            self.ds_val,
            batch_size=self.batch_size_val,
            shuffle=self.shuffle_val,
            num_workers=self.num_workers_val,
            collate_fn=self.collate_fn_val,
        )

    def test_dataloader(self):
        if self.ds_train is None:
            raise ValueError(
                "'dir_test' not provided, test dataset not initialized."
            )
        return DataLoader(
            self.ds_train,
            batch_size=self.batch_size_test,
            shuffle=self.shuffle_test,
            num_workers=self.num_workers_test,
            collate_fn=self.collate_fn_test,
        )
