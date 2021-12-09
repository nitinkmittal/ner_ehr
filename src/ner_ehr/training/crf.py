from typing import Optional

import torch
import torch.nn as nn


class CRF(nn.Module):
    def __init__(
        self,
        num_classes: int,
    ) -> None:
        """ """
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
                ``(seq_length, batch_size, num_classes)``
                if ``batch_first`` is ``False``,
                ``(batch_size, seq_length, num_classes)`` otherwise

            tags (`~torch.LongTensor`): Sequence of tags tensor of size
                ``(seq_length, batch_size)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length)`` otherwise

            masks (`~torch.BoolTensor`): Mask tensor of
                size ``(seq_length, batch_size)``
                if ``batch_first`` is ``False``,
                ``(batch_size, seq_length)`` otherwise

            reduction: Specifies  the reduction to apply to the output:
                ``none|sum|mean|token_mean``.
                ``none``: no reduction will be applied.
                ``sum``: the output will be summed over batches.
                ``mean``: the output will be
                averaged over batches. ``token_mean``:
                the output will be averaged over tokens

        Returns:
            `~torch.Tensor`: The log likelihood.
            This will have size ``(batch_size,)`` if
            reduction is ``none``, ``()`` otherwise.
        """

        emissions = emissions.transpose(0, 1)
        tags = tags.transpose(0, 1)
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
        """ """
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
        """ """

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

            # Set score to the next score if this timestep is valid (masks == 1)
            # shape: (batch_size, num_classes)
            score = torch.where(masks[i].unsqueeze(1), next_score, score)

        # End transition score
        # shape: (batch_size, num_classes)
        score += self.end_transitions

        # Sum (log-sum-exp) over all possible tags
        # shape: (batch_size,)
        return torch.logsumexp(score, dim=1)
