from torch import nn, Tensor
from typing import Optional


class LSTMNERTagger(nn.Module):
    """LSTM for NER"""

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
