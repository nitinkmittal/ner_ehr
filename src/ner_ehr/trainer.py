import pytorch_lightning as pl

from typing import List, Union, Optional
from pathlib import Path
import torch
from torch import Tensor
from .metrics import dice_loss, bce_loss, accuracy

from .unet import UNet
import os
from .utils import colorize_mask
from torchvision.transforms import ToPILImage
import numpy as np
from torchvision.utils import make_grid
from torch.nn import BCELoss

from ner_ehr.models import LSTMNERTagger


class LitLSTMNERTagger(pl.LightningModule):
    def __init__(
        self,
        embedding_dim: int,
        vocab_size: int,
        hidden_size: int,
        num_classes: int,
        embedding_weights: Tensor = Optional[None],
        num_lstm_layers: int = 1,
        lstm_dropout: float = 0,
        bidirectional: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters()
        self.lstm = LSTMNERTagger(
            embedding_dim=embedding_dim,
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_classes=num_classes,
            embedding_weights=embedding_weights,
            num_lstm_layers=num_lstm_layers,
            lstm_dropout=lstm_dropout,
            bidirectional=bidirectional,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.lstm(x)

    def training_step(self, train_batch, batch_idx):
        fps, imgs, masks = train_batch
        pred_masks = self.forward(imgs)
        pred_masks = torch.sigmoid(pred_masks)

        dice = dice_loss(pred_mask=pred_masks, true_mask=masks)
        bce = self.bce_loss(pred_masks, masks)
        total = (
            self.hparams.dice_coeff * dice
            + (1 - self.hparams.dice_coeff) * bce
        )
        acc = accuracy(pred_mask=pred_masks, true_mask=masks)

        self.log(
            "train_dice_loss",
            dice.item(),
            on_step=True,
            on_epoch=True,
        )
        self.log(
            "train_bce_loss",
            bce.item(),
            on_step=True,
            on_epoch=True,
        )

        self.log(
            "train_total_loss",
            total.item(),
            on_step=True,
            on_epoch=True,
        )
        self.log(
            "train_acc",
            acc.item(),
            on_step=True,
            on_epoch=True,
        )

        return total

    def _save_img(
        self, fp: Union[Path, str], pred_mask: Tensor, true_mask: Tensor
    ) -> None:

        pred_mask = torch.argmax(pred_mask, dim=0, keepdim=True)
        pred_mask = colorize_mask(
            pred_mask.cpu(), num_classes=self.hparams.out_channels
        )

        true_mask = torch.argmax(true_mask, dim=0, keepdim=True)
        true_mask = colorize_mask(
            true_mask.cpu(), num_classes=self.hparams.out_channels
        )

        fp = ".".join(fp.split(".")[:-1])
        if not os.path.isdir(self.hparams.val_mask_dir):
            os.makedirs(self.hparams.val_mask_dir, exist_ok=True)
        fp = os.path.join(
            self.hparams.val_mask_dir, f"{fp}_step={self.global_step}.png"
        )

        ToPILImage()(make_grid([true_mask, pred_mask], pad_value=1.0)).save(fp)

    def validation_step(self, val_batch, batch_idx):
        fps, imgs, masks = val_batch

        pred_masks = self.forward(imgs)
        pred_masks = torch.sigmoid(pred_masks)
        dice = dice_loss(pred_mask=pred_masks, true_mask=masks)
        bce = self.bce_loss(pred_masks, masks)
        total = (
            self.hparams.dice_coeff * dice
            + (1 - self.hparams.dice_coeff) * bce
        )
        acc = accuracy(pred_mask=pred_masks, true_mask=masks)

        self.log(
            "val_dice_loss",
            dice.item(),
            on_step=True,
            on_epoch=True,
        )
        self.log(
            "val_bce_loss",
            bce.item(),
            on_step=True,
            on_epoch=True,
        )

        self.log(
            "val_total_loss",
            total.item(),
            on_step=True,
            on_epoch=True,
        )
        self.log(
            "val_acc",
            acc.item(),
            on_step=True,
            on_epoch=True,
        )

        for i, fp in enumerate(fps):
            if np.random.uniform() > 0.9:
                self._save_img(fp, pred_masks[i], masks[i])

        return total

    def configure_optimizers(self):
        return torch.optim.Adam(
            params=self.unet.parameters(), lr=self.hparams.lr
        )
