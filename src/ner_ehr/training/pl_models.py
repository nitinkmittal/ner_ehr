import os
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch

from ner_ehr.data import Constants
from ner_ehr.training import metrics
from ner_ehr.training.losses import cross_entropy
from ner_ehr.training.models import (
    BIDIRECTIONAL,
    LSTM_DROPOUT,
    NUM_LSTM_LAYERS,
    LSTMCRFNERTagger,
    LSTMNERTagger,
)
from ner_ehr.utils import load_np, save_np

EMBEDDING_WEIGHTS_FP: Optional[Union[str, Path]] = None
LR: float = 0.001
DTYPE = None
SAVE_CM_AFTER_EVERY_N_EPOCHS: int = 1
USE_MASKS: bool = False
CE_WEIGHT: float = 1.0
CRF_NLLH_WEIGHT: float = 0.001


class LitLSTMNERTagger(pl.LightningModule):
    def __init__(
        self,
        embedding_dim: int,
        vocab_size: int,
        hidden_size: int,
        num_classes: int,
        embedding_weights_fp: Optional[
            Union[str, Path]
        ] = EMBEDDING_WEIGHTS_FP,
        num_lstm_layers: int = NUM_LSTM_LAYERS,
        lstm_dropout: float = LSTM_DROPOUT,
        bidirectional: bool = BIDIRECTIONAL,
        lr: float = LR,
        dtype: Optional = DTYPE,
        save_cm_after_every_n_epochs: int = SAVE_CM_AFTER_EVERY_N_EPOCHS,
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters()

        embedding_weights: torch.FloatTensor = None
        # load per-trained embeddings if given
        if embedding_weights_fp is not None:
            embedding_weights = load_np(fp=embedding_weights_fp)
            embedding_weights = torch.tensor(
                embedding_weights, dtype=dtype
            ).cuda()

        # model initialization
        self.lstm_ner_tagger = LSTMNERTagger(
            embedding_dim=embedding_dim,
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_classes=num_classes,
            embedding_weights=embedding_weights,
            num_lstm_layers=num_lstm_layers,
            lstm_dropout=lstm_dropout,
            bidirectional=bidirectional,
        )
        self._init_cm_train()
        self._init_cm_val()

    def _init_cm_train(
        self,
    ) -> None:
        """Initialize confusion matrix for training labels."""
        self.cm_argmax_train: np.ndarray = np.zeros(
            (self.hparams.num_classes, self.hparams.num_classes), dtype=np.int
        )

    def _init_cm_val(
        self,
    ) -> None:
        """Initialize confusion matrix for validation labels."""
        self.cm_argmax_val: np.ndarray = np.zeros(
            (self.hparams.num_classes, self.hparams.num_classes), dtype=np.int
        )

    def forward(self, X: torch.FloatTensor) -> torch.FloatTensor:
        return self.lstm_ner_tagger(X)

    def on_train_epoch_end(
        self,
    ):
        """Hook after training epoch end."""
        if (
            self.current_epoch == 0
            or (self.current_epoch + 1)
            % self.hparams.save_cm_after_every_n_epochs
            == 0
        ):
            save_np(
                arr=self.cm_argmax_train,
                fp=os.path.join(
                    self.logger.log_dir,
                    f"train_cm_argmax_epoch={self.current_epoch}.npy",
                ),
            )
        self._init_cm_train()

    def validation_epoch_end(self, val_step_end_outputs: torch.FloatTensor):
        """Hook after validation epoch end."""
        if (
            self.current_epoch == 0
            or (self.current_epoch + 1)
            % self.hparams.save_cm_after_every_n_epochs
            == 0
        ):
            save_np(
                arr=self.cm_argmax_val,
                fp=os.path.join(
                    self.logger.log_dir,
                    f"val_cm_argmax_epoch={self.current_epoch}.npy",
                ),
            )
        self._init_cm_val()

    def training_step(self, batch, batch_idx: int) -> float:
        X, Y, _ = batch
        Y_hat = self.forward(X)

        ce_loss = cross_entropy(
            Y_hat=Y_hat,
            Y=Y,
            ignore_index=Constants.PAD_TOKEN_ENTITY_INT_LABEL.value,
        )

        argmax_acc, argmax_accs, argmax_cm = metrics.all_metrics(
            Y_hat=Y_hat,
            Y=Y,
            decode_mode="argmax",
        )
        self.cm_argmax_train += argmax_cm

        self.log(
            "train_ce_loss",
            ce_loss.item(),
            on_step=True,
            on_epoch=True,
            batch_size=len(X),
        )
        self.log(
            "train_argmax_acc",
            argmax_acc,
            on_step=True,
            on_epoch=True,
            batch_size=len(X),
        )
        for label, acc in enumerate(argmax_accs):
            self.log(
                f"train_argmax_acc_label={label}",
                argmax_acc,
                on_step=True,
                on_epoch=True,
                batch_size=len(X),
            )

        return ce_loss

    def validation_step(self, batch, batch_idx: int) -> float:
        X, Y, _ = batch
        Y_hat = self.forward(X)

        ce_loss = cross_entropy(
            Y_hat=Y_hat,
            Y=Y,
            ignore_index=Constants.PAD_TOKEN_ENTITY_INT_LABEL.value,
        )

        argmax_acc, argmax_accs, argmax_cm = metrics.all_metrics(
            Y_hat=Y_hat,
            Y=Y,
            decode_mode="argmax",
        )
        self.cm_argmax_val += argmax_cm

        self.log(
            "val_loss",
            ce_loss.item(),
            on_step=True,
            on_epoch=True,
            batch_size=len(X),
        )
        self.log(
            "val_argmax_acc",
            argmax_acc,
            on_step=True,
            on_epoch=True,
            batch_size=len(X),
        )
        for label, acc in enumerate(argmax_accs):
            self.log(
                f"val_argmax_acc_label={label}",
                argmax_acc,
                on_step=True,
                on_epoch=True,
                batch_size=len(X),
            )

        return ce_loss

    def configure_optimizers(self):
        return torch.optim.Adam(
            params=self.lstm_ner_tagger.parameters(), lr=self.hparams.lr
        )


class LitLSTMCRFNERTagger(pl.LightningModule):
    def __init__(
        self,
        embedding_dim: int,
        vocab_size: int,
        hidden_size: int,
        num_classes: int,
        embedding_weights_fp: Optional[
            Union[str, Path]
        ] = EMBEDDING_WEIGHTS_FP,
        num_lstm_layers: int = NUM_LSTM_LAYERS,
        lstm_dropout: float = LSTM_DROPOUT,
        bidirectional: bool = BIDIRECTIONAL,
        use_masks: bool = USE_MASKS,
        ce_weight: float = CE_WEIGHT,
        crf_nllh_weight: float = CRF_NLLH_WEIGHT,
        lr: float = LR,
        dtype: Optional = DTYPE,
        save_cm_after_every_n_epochs: int = SAVE_CM_AFTER_EVERY_N_EPOCHS,
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters()

        embedding_weights: torch.FloatTensor = None
        # load per-trained embeddings if given
        if embedding_weights_fp is not None:
            embedding_weights = load_np(fp=embedding_weights_fp)
            embedding_weights = torch.tensor(
                embedding_weights, dtype=dtype
            ).cuda()

        # model initialization
        self.lstm_crf_ner_tagger = LSTMCRFNERTagger(
            embedding_dim=embedding_dim,
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_classes=num_classes,
            embedding_weights=embedding_weights,
            num_lstm_layers=num_lstm_layers,
            lstm_dropout=lstm_dropout,
            bidirectional=bidirectional,
        )
        self._init_cm_train()
        self._init_cm_val()

    def _init_cm_train(
        self,
    ) -> None:
        """Initialize confusion matrix for training labels."""
        self.cm_argmax_train: np.ndarray = np.zeros(
            (self.hparams.num_classes, self.hparams.num_classes), dtype=np.int
        )
        self.cm_viterbi_train: np.ndarray = np.zeros(
            (self.hparams.num_classes, self.hparams.num_classes), dtype=np.int
        )

    def _init_cm_val(
        self,
    ) -> None:
        """Initialize confusion matrix for validation labels."""
        self.cm_argmax_val: np.ndarray = np.zeros(
            (self.hparams.num_classes, self.hparams.num_classes), dtype=np.int
        )
        self.cm_viterbi_val: np.ndarray = np.zeros(
            (self.hparams.num_classes, self.hparams.num_classes), dtype=np.int
        )

    def forward(
        self,
        X: torch.FloatTensor,
        Y: torch.LongTensor,
        masks: Optional[torch.BoolTensor] = None,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """ """
        return self.lstm_crf_ner_tagger(X=X, Y=Y, masks=masks)

    def on_train_epoch_end(
        self,
    ):
        """Hook after training epoch end."""
        if (
            self.current_epoch == 0
            or (self.current_epoch + 1)
            % self.hparams.save_cm_after_every_n_epochs
            == 0
        ):
            save_np(
                arr=self.cm_argmax_train,
                fp=os.path.join(
                    self.logger.log_dir,
                    f"train_cm_argmax_epoch={self.current_epoch}.npy",
                ),
            )
            save_np(
                arr=self.cm_viterbi_train,
                fp=os.path.join(
                    self.logger.log_dir,
                    f"train_cm_viterbi_epoch={self.current_epoch}.npy",
                ),
            )
        self._init_cm_train()

    def validation_epoch_end(self, val_step_end_outputs: torch.FloatTensor):
        """Hook after validation epoch end."""
        if (
            self.current_epoch == 0
            or (self.current_epoch + 1)
            % self.hparams.save_cm_after_every_n_epochs
            == 0
        ):
            save_np(
                arr=self.cm_argmax_val,
                fp=os.path.join(
                    self.logger.log_dir,
                    f"val_cm_argmax_epoch={self.current_epoch}.npy",
                ),
            )
            save_np(
                arr=self.cm_viterbi_val,
                fp=os.path.join(
                    self.logger.log_dir,
                    f"val_cm_viterbi_epoch={self.current_epoch}.npy",
                ),
            )
        self._init_cm_val()

    def training_step(self, batch, batch_idx: int) -> float:
        X, Y, _ = batch
        masks: torch.FloatTensor = None
        if self.hparams.use_masks:
            masks = torch.ones_like(Y, dtype=torch.bool)
            masks = torch.where(
                Y == Constants.PAD_TOKEN_ENTITY_INT_LABEL.value, False, masks
            )

        crf_nllh, Y_hat = self.forward(X=X, Y=Y, masks=masks)
        crf_nllh *= self.hparams.crf_nllh_weight

        ce_loss = self.hparams.ce_weight * cross_entropy(
            Y_hat=Y_hat,
            Y=Y,
            ignore_index=Constants.PAD_TOKEN_ENTITY_INT_LABEL.value,
        )

        total_loss = ce_loss + crf_nllh

        argmax_acc, argmax_accs, argmax_cm = metrics.all_metrics(
            Y_hat=Y_hat,
            Y=Y,
            decode_mode="argmax",
        )
        self.cm_argmax_train += argmax_cm

        viterbi_acc, viterbi_accs, viterbi_cm = metrics.all_metrics(
            Y_hat=Y_hat,
            Y=Y,
            decode_mode="viterbi",
            start_transitions=self.lstm_crf_ner_tagger.crf.start_transitions.data,
            end_transitions=self.lstm_crf_ner_tagger.crf.end_transitions.data,
            transitions=self.lstm_crf_ner_tagger.crf.transitions.data,
            masks=masks,
        )
        self.cm_viterbi_train += viterbi_cm

        self.log(
            "train_ce_loss",
            ce_loss.item(),
            on_step=True,
            on_epoch=True,
            batch_size=len(X),
        )
        self.log(
            "train_crf_nllh",
            crf_nllh.item(),
            on_step=True,
            on_epoch=True,
            batch_size=len(X),
        )
        self.log(
            "train_loss",
            total_loss.item(),
            on_step=True,
            on_epoch=True,
            batch_size=len(X),
        )

        self.log(
            "train_argmax_acc",
            argmax_acc,
            on_step=True,
            on_epoch=True,
            batch_size=len(X),
        )
        self.log(
            "train_viterbi_acc",
            viterbi_acc,
            on_step=True,
            on_epoch=True,
            batch_size=len(X),
        )
        for label, (argmax_acc, viterbi_acc) in enumerate(
            zip(argmax_accs, viterbi_accs)
        ):
            self.log(
                f"train_argmax_acc_label={label}",
                argmax_acc,
                on_step=True,
                on_epoch=True,
                batch_size=len(X),
            )
            self.log(
                f"train_viterbi_acc_label={label}",
                viterbi_acc,
                on_step=True,
                on_epoch=True,
                batch_size=len(X),
            )

        return total_loss

    def validation_step(self, batch, batch_idx: int) -> float:
        X, Y, _ = batch
        masks: torch.FloatTensor = None
        if self.hparams.use_masks:
            masks = torch.ones_like(Y, dtype=torch.bool)
            masks = torch.where(
                Y == Constants.PAD_TOKEN_ENTITY_INT_LABEL.value, False, masks
            )

        crf_nllh, Y_hat = self.forward(X=X, Y=Y, masks=masks)
        crf_nllh *= self.hparams.crf_nllh_weight

        ce_loss = self.hparams.ce_weight * cross_entropy(
            Y_hat=Y_hat,
            Y=Y,
            ignore_index=Constants.PAD_TOKEN_ENTITY_INT_LABEL.value,
        )

        total_loss = ce_loss + crf_nllh

        argmax_acc, argmax_accs, argmax_cm = metrics.all_metrics(
            Y_hat=Y_hat,
            Y=Y,
            decode_mode="argmax",
        )
        self.cm_argmax_val += argmax_cm

        viterbi_acc, viterbi_accs, viterbi_cm = metrics.all_metrics(
            Y_hat=Y_hat,
            Y=Y,
            decode_mode="viterbi",
            start_transitions=self.lstm_crf_ner_tagger.crf.start_transitions.data,
            end_transitions=self.lstm_crf_ner_tagger.crf.end_transitions.data,
            transitions=self.lstm_crf_ner_tagger.crf.transitions.data,
            masks=masks,
        )
        self.cm_viterbi_val += viterbi_cm

        self.log(
            "val_ce_loss",
            ce_loss.item(),
            on_step=True,
            on_epoch=True,
            batch_size=len(X),
        )
        self.log(
            "val_crf_nllh",
            crf_nllh.item(),
            on_step=True,
            on_epoch=True,
            batch_size=len(X),
        )
        self.log(
            "val_loss",
            total_loss.item(),
            on_step=True,
            on_epoch=True,
            batch_size=len(X),
        )

        self.log(
            "val_argmax_acc",
            argmax_acc,
            on_step=True,
            on_epoch=True,
            batch_size=len(X),
        )
        self.log(
            "val_viterbi_acc",
            viterbi_acc,
            on_step=True,
            on_epoch=True,
            batch_size=len(X),
        )
        for label, (argmax_acc, viterbi_acc) in enumerate(
            zip(argmax_accs, viterbi_accs)
        ):
            self.log(
                f"val_argmax_acc_label={label}",
                argmax_acc,
                on_step=True,
                on_epoch=True,
                batch_size=len(X),
            )
            self.log(
                f"val_viterbi_acc_label={label}",
                viterbi_acc,
                on_step=True,
                on_epoch=True,
                batch_size=len(X),
            )

        return total_loss

    def configure_optimizers(self):
        return torch.optim.Adam(
            params=self.lstm_crf_ner_tagger.parameters(),
            lr=self.hparams.lr,
        )
