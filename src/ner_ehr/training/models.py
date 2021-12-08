import os
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pytorch_lightning as pl
import torch
from torch import Tensor, nn

from ner_ehr.data import Constants
from ner_ehr.training import metrics
from ner_ehr.training.losses import cross_entropy
from ner_ehr.utils import load_np, save_np


class LSTMNERTagger(nn.Module):
    """LSTM for NER"""

    def __init__(
        self,
        embedding_dim: int,
        vocab_size: int,
        hidden_size: int,
        num_classes: int,
        embedding_weights: Optional[Tensor] = None,
        num_lstm_layers: int = 1,
        lstm_dropout: float = 0,
        bidirectional: bool = False,
    ):
        """
        Args:
            embedding_dim: embedding dimension

            vocab_size: vocabulary size

            hidden_size: number of hidden units in LSTM layer

            num_classes: number of classes/entities to tag

            embedding_weights: pre-trained embeddings weights
        """
        super().__init__()
        if embedding_weights is not None:
            self.embed = nn.Embedding.from_pretrained(embedding_weights)
        else:
            self.embed = nn.Embedding(
                num_embeddings=vocab_size, embedding_dim=embedding_dim
            )

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            bidirectional=bidirectional,
            batch_first=True,
        )

        self.fc1 = nn.Linear(
            in_features=2 * hidden_size if bidirectional else hidden_size,
            out_features=num_classes,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass over input tensor.

        Args:
            x: IntTensor or LongTensor of shape (B, S)
                B: batch_size

                S: seq_length

        Returns:
            FloatTensor of shape (B, S, num_classes)
        """
        x = self.embed(x)  # (B, S) -> (B, S, embedding_dim)
        x, (_, _) = self.lstm(
            x
        )  # (B, S, embedding_dim) -> (B, S, hidden_size)
        x = self.fc1(x)  # (B, S, hidden_size) -> (B, S, num_classes)
        return x


class LitLSTMNERTagger(pl.LightningModule):
    def __init__(
        self,
        embedding_dim: int,
        vocab_size: int,
        hidden_size: int,
        num_classes: int,
        embedding_weights_fp: Optional[Union[str, Path]] = None,
        num_lstm_layers: int = 1,
        lstm_dropout: float = 0,
        bidirectional: bool = False,
        lr: float = 0.001,
        dtype: Optional = None,
        save_cm_after_every_n_epochs: int = 1,
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters()

        embedding_weights: Tensor = None
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
        self.cm_train: np.ndarray = np.zeros(
            (self.hparams.num_classes, self.hparams.num_classes), dtype=np.int
        )

    def _init_cm_val(
        self,
    ) -> None:
        """Initialize confusion matrix for validation labels."""
        self.cm_val: np.ndarray = np.zeros(
            (self.hparams.num_classes, self.hparams.num_classes), dtype=np.int
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.lstm_ner_tagger(x)

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
                arr=self.cm_train,
                fp=os.path.join(
                    self.logger.log_dir,
                    f"train_cm_epoch={self.current_epoch}.npy",
                ),
            )
        self._init_cm_train()

    def validation_epoch_end(self, val_step_end_outputs: Tensor):
        """Hook after validation epoch end."""
        if (
            self.current_epoch == 0
            or (self.current_epoch + 1)
            % self.hparams.save_cm_after_every_n_epochs
            == 0
        ):
            save_np(
                arr=self.cm_val,
                fp=os.path.join(
                    self.logger.log_dir,
                    f"val_cm_epoch={self.current_epoch}.npy",
                ),
            )
        self._init_cm_val()

    def training_step(self, batch, batch_idx: int) -> float:
        X, Y, _ = batch
        Y_hat = self.forward(X)

        loss = cross_entropy(
            Y_hat=Y_hat,
            Y=Y,
            ignore_index=Constants.PAD_TOKEN_ENTITY_INT_LABEL.value,
        )

        acc, accs, cm = metrics.all_metrics(Y_hat=Y_hat, Y=Y)
        self.cm_train += cm

        self.log(
            "train_loss",
            loss.item(),
            on_step=True,
            on_epoch=True,
            batch_size=len(X),
        )
        self.log(
            "train_acc",
            acc,
            on_step=True,
            on_epoch=True,
            batch_size=len(X),
        )
        for label, acc in enumerate(accs):
            self.log(
                f"train_acc_label={label}",
                acc,
                on_step=True,
                on_epoch=True,
                batch_size=len(X),
            )

        return loss

    def validation_step(self, batch, batch_idx: int) -> float:
        X, Y, meta = batch
        Y_hat = self.forward(X)

        loss = cross_entropy(
            Y_hat=Y_hat,
            Y=Y,
            ignore_index=Constants.PAD_TOKEN_ENTITY_INT_LABEL.value,
        )

        acc, accs, cm = metrics.all_metrics(Y_hat=Y_hat, Y=Y)
        self.cm_val += cm

        self.log(
            "val_loss",
            loss.item(),
            on_step=True,
            on_epoch=True,
            batch_size=len(X),
        )
        self.log(
            "val_acc",
            acc,
            on_step=True,
            on_epoch=True,
            batch_size=len(X),
        )
        for label, acc in enumerate(accs):
            self.log(
                f"val_acc_label={label}",
                acc,
                on_step=True,
                on_epoch=True,
                batch_size=len(X),
            )

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(
            params=self.lstm_ner_tagger.parameters(), lr=self.hparams.lr
        )
