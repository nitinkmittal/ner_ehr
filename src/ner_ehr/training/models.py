from typing import Optional, Tuple

import torch
from torch import Tensor, nn

from ner_ehr.training.crf import CRF

NUM_LSTM_LAYERS: int = 1
LSTM_DROPOUT: float = 0.1
BIDIRECTIONAL: bool = False
EMBEDDING_WEIGHTS: Optional[torch.FloatTensor] = None


class LSTMNERTagger(nn.Module):
    """LSTM for NER"""

    def __init__(
        self,
        embedding_dim: int,
        vocab_size: int,
        hidden_size: int,
        num_classes: int,
        embedding_weights: Optional[torch.FloatTensor] = EMBEDDING_WEIGHTS,
        num_lstm_layers: int = NUM_LSTM_LAYERS,
        lstm_dropout: float = LSTM_DROPOUT,
        bidirectional: bool = BIDIRECTIONAL,
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

    def forward(self, X: Tensor) -> Tensor:
        """Forward pass over input tensor.

        Args:
            X: IntTensor or LongTensor of shape (B, S)
                B: batch_size

                S: seq_length

        Returns:
            FloatTensor of shape (B, S, num_classes)
        """
        X = self.embed(X)  # (B, S) -> (B, S, embedding_dim)
        X, (_, _) = self.lstm(
            X
        )  # (B, S, embedding_dim) -> (B, S, hidden_size)
        X = self.fc1(X)  # (B, S, hidden_size) -> (B, S, num_classes)
        return X


class LSTMCRFNERTagger(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        vocab_size: int,
        hidden_size: int,
        num_classes: int,
        embedding_weights: Optional[torch.FloatTensor] = EMBEDDING_WEIGHTS,
        num_lstm_layers: int = NUM_LSTM_LAYERS,
        lstm_dropout: float = LSTM_DROPOUT,
        bidirectional: bool = BIDIRECTIONAL,
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
        self.crf = CRF(num_classes=num_classes)

    def forward(
        self,
        X: torch.FloatTensor,
        Y: torch.LongTensor,
        masks: Optional[torch.BoolTensor] = None,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        Y_hat = self.lstm_ner_tagger(X)
        crf_llh = self.crf(
            emissions=Y_hat,
            tags=Y,
            masks=masks,
        )
        return -crf_llh, Y_hat
