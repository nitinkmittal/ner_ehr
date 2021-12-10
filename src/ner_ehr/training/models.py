from typing import Optional, Tuple

import torch
from torch import nn
from ner_ehr.utils import copy_docstring

NUM_LSTM_LAYERS: int = 1
LSTM_DROPOUT: float = 0.0
BIDIRECTIONAL: bool = False
EMBEDDING_WEIGHTS: Optional[torch.FloatTensor] = None


class CRF(nn.Module):
    """Conditional random field.
    This module implements a conditional random field [LMP01]_.
    The forward computation of this class computes the log likelihood
    of the given sequence of tags and emission score tensor.
    This class also has `~CRF.decode` method which finds
    the best tag sequence given an emission score tensor
    using `Viterbi algorithm`_.

    Args:
        num_tags: Number of tags.
        batch_first: Whether the first dimension corresponds
            to the size of a minibatch.

    Attributes:
        start_transitions (`~torch.nn.Parameter`): Start transition
            score tensor of size ``(num_tags,)``.
        end_transitions (`~torch.nn.Parameter`): End transition
            score tensor of size ``(num_tags,)``.
        transitions (`~torch.nn.Parameter`): Transition score
            tensor of size ``(num_tags, num_tags)``.
    .. [LMP01] Lafferty, J., McCallum, A., Pereira, F. (2001).
       "Conditional random fields: Probabilistic models for segmenting and
       labeling sequence data". *Proc. 18th International Conf. on Machine
       Learning*. Morgan Kaufmann. pp. 282–289.
    .. _Viterbi algorithm: https://en.wikipedia.org/wiki/Viterbi_algorithm

    Code copied from https://github.com/kmkurn/pytorch-crf/blob/8f3203a1f1d7984c87718bfe31853242670258db/torchcrf/__init__.py
    Original author: https://github.com/kmkurn
    """

    def __init__(
        self,
        num_classes: int,
    ) -> None:
        """
        Args:
            num_classes: Number of classes/tags to classify/tag among
        """
        super().__init__()
        self.num_classes = num_classes
        self.start_transitions: torch.FloatTensor = nn.Parameter(
            torch.empty(num_classes)
        )
        self.end_transitions: torch.FloatTensor = nn.Parameter(
            torch.empty(num_classes)
        )
        self.transitions: torch.FloatTensor = nn.Parameter(
            torch.empty(num_classes, num_classes)
        )
        self._init_parameters()

    def _init_parameters(self) -> None:
        """Initialize the transition parameters.
        The parameters will be initialized randomly from a uniform distribution
        between -0.1 and 0.1.
        """
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)
        nn.init.uniform_(self.transitions, -0.1, 0.1)

    def forward(
        self,
        emissions: torch.FloatTensor,
        tags: torch.LongTensor,
        masks: torch.BoolTensor,
    ) -> torch.Tensor:
        """Compute the conditional log likelihood of a sequence of tags
            given emission scores.

        Args:
            emissions (`~torch.Tensor`): Emission score tensor of size
                ``(batch_size, seq_length, num_classes)``

            tags (`~torch.LongTensor`): Sequence of tags tensor of size
                ``(batch_size, seq_length)``

            masks (`~torch.BoolTensor`): Mask tensor of size
                ``(batch_size, seq_length)``

        Returns:
            The float (torch.FloatTensor) type mean log-likelihood
        """
        emissions = emissions.transpose(0, 1)
        tags = tags.transpose(0, 1)
        if masks is None:
            masks = torch.ones_like(tags, dtype=torch.bool)
        else:
            masks = masks.transpose(0, 1)

        # shape: (batch_size,)
        numerator = self._compute_score(emissions, tags, masks)
        # shape: (batch_size,)
        denominator = self._compute_normalizer(emissions, masks)
        # shape: (batch_size,)
        llh = numerator - denominator
        return llh.mean()

    def _compute_score(
        self,
        emissions: torch.Tensor,
        tags: torch.LongTensor,
        masks: torch.BoolTensor,
    ) -> torch.FloatTensor:
        """
        Args:
            emissions (`~torch.Tensor`): Emission score tensor of size
                ``(seq_length, batch_size, num_classes)``

            tags (`~torch.LongTensor`): Sequence of tags tensor of size
                ``(seq_length, batch_size)``

            masks (`~torch.BoolTensor`): Mask tensor of size
                ``(seq_length, batch_size)``

        Returns:
            Numerator value for forward pass of size ``(batch_size, )``

        TODO: Update/Improve docstring
        """
        seq_length, batch_size = tags.shape
        masks = masks.float()

        # Start transition score and first emission
        # shape: (batch_size,)
        score = self.start_transitions[tags[0]]
        score += emissions[0, torch.arange(batch_size), tags[0]]

        for i in range(1, seq_length):
            # Transition score to next tag,
            #   only added if next timestep is valid (masks == 1)
            # shape: (batch_size,)
            score += self.transitions[tags[i - 1], tags[i]] * masks[i]
            # Emission score for next tag,
            #   only added if next timestep is valid (masks == 1)
            # shape: (batch_size,)
            score += emissions[i, torch.arange(batch_size), tags[i]] * masks[i]

        # End transition score
        # shape: (batch_size,)
        seq_ends = masks.long().sum(dim=0) - 1
        # shape: (batch_size,)
        last_tags = tags[seq_ends, torch.arange(batch_size)]
        # shape: (batch_size,)
        score += self.end_transitions[last_tags]

        return score

    def _compute_normalizer(
        self, emissions: torch.Tensor, masks: torch.BoolTensor
    ) -> torch.FloatTensor:
        """
        Args:
            emissions (`~torch.Tensor`): Emission score tensor of size
                ``(seq_length, batch_size, num_classes)``

            masks (`~torch.BoolTensor`): Mask tensor of size
                ``(seq_length, batch_size)``

        Returns:
            Denominator value for forward pass of size ``(batch_size, )``

        TODO: Update/Improve docstring
        """

        seq_length = emissions.size(0)

        # Start transition score and first emission; score has size of
        # (batch_size, num_classes) where for each batch, the j-th column
        # stores the score that the first timestep has tag j
        # shape: (batch_size, num_classes)
        score = self.start_transitions + emissions[0]

        for i in range(1, seq_length):
            # Broadcast score for every possible next tag
            # shape: (batch_size, num_classes, 1)
            broadcast_score = score.unsqueeze(2)
            # Broadcast emission score for every possible current tag
            # shape: (batch_size, 1, num_classes)
            broadcast_emissions = emissions[i].unsqueeze(1)
            # Compute the score tensor of size
            #   (batch_size, num_classes, num_classes) where
            #   for each sample, entry at row i and column j
            #   stores the sum of scores of all
            #   possible tag sequences so far that end with
            #   transitioning from tag i to tag j and emitting
            # shape: (batch_size, num_classes, num_classes)
            next_score = (
                broadcast_score + self.transitions + broadcast_emissions
            )
            # Sum over all possible current tags,
            #   but we're in score space, so a sum becomes
            #   a log-sum-exp: for each sample, entry i stores the
            #   sum of scores of all possible tag sequences so far,
            #   that end in tag i
            # shape: (batch_size, num_classes)
            next_score = torch.logsumexp(next_score, dim=1)

            # Set score to the next score if this timestep is valid
            #   (masks == 1)
            # shape: (batch_size, num_classes)
            score = torch.where(masks[i].unsqueeze(1), next_score, score)

        # End transition score
        # shape: (batch_size, num_classes)
        score += self.end_transitions

        # Sum (log-sum-exp) over all possible tags
        # shape: (batch_size,)
        return torch.logsumexp(score, dim=1)


class LSTMNERTagger(nn.Module):
    """LSTM for Name Entity Recognization."""

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

            embedding_weights: pre-trained embeddings weights,
                optional, default=None
                if None, embedding weights are learned on the fly,
                if not None, then embedding layer is set to untrainable

            num_lstm_layers: number of recurrent layers
                E.g., setting num_layers=2 would mean stacking two LSTMs
                together to form a stacked LSTM, with the second LSTM
                taking in outputs of the first LSTM and
                computing the final results, default=1

            lstm_dropout: if non-zero, introduces a dropout layer
                on the outputs of each LSTM layer except the last layer,
                with dropout probability equal to dropout. default=0.0

            bidirectional: If True, becomes a bidirectional LSTM
                default=False

        Note: `batch_first` is set to True for LSTM layer by default
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

    def forward(self, X: torch.LongTensor) -> torch.FloatTensor:
        """Forward pass for LSTMNERTagger.

        Args:
            X: tensor of shape (batch_size, seq_length)

        Returns:
            A tensor of shape (batch_size, seq_length, num_classes)
        """
        X = self.embed(X)  # (B, S) -> (B, S, embedding_dim)
        X, (_, _) = self.lstm(
            X
        )  # (B, S, embedding_dim) -> (B, S, hidden_size)
        X = self.fc1(X)  # (B, S, hidden_size) -> (B, S, num_classes)
        return X


class LSTMCRFNERTagger(nn.Module):
    """LSTM with CRF for Name Entity Recognization."""

    @copy_docstring(LSTMNERTagger.__init__)
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
        """Forward pass for LSTMCRFNERTagger.

        Args:
            X: tensor of shape (batch_size, seq_length)

            Y: tensor of shape (batch_size, seq_length)

            masks: tensor of shape (batch_size, seq_length), optional
                default=None

        Returns:
            A tuple of scalar tensor with negative log-likelihood
                from conditional random field and
                a float tensor of shape (batch_size, seq_length, num_classes)
        """
        Y_hat = self.lstm_ner_tagger(X)
        crf_llh = self.crf(
            emissions=Y_hat,
            tags=Y,
            masks=masks,
        )
        return -crf_llh, Y_hat
