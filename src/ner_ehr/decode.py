"""This module can be used to decode emissions
    using argmax and viterbi decoding techniques."""
from typing import Optional

import torch
from torch import nn

from ner_ehr.data import Constants


def argmax_decode(emissions: torch.FloatTensor) -> torch.LongTensor:
    """Greedy decoding of state/tag/entity sequence.

    Args:
        emissions `torch.FloatTensor`: (batch_size, seq_length, num_classes)

    Returns:
        sequences `torch.LongTensor`: (batch_size, seq_length)
    """
    assert emissions.ndim == 3
    with torch.no_grad():
        seqs = torch.argmax(nn.functional.softmax(emissions, dim=-1), dim=-1)
    return seqs


def viterbi_decode(
    emissions: torch.FloatTensor,
    start_transitions: torch.FloatTensor,
    end_transitions: torch.FloatTensor,
    transitions: torch.FloatTensor,
    masks: Optional[torch.BoolTensor] = None,
) -> torch.LongTensor:
    """
    Viterbi decoding of state/tag/entity sequence.

    Copied from: https://github.com/kmkurn/pytorch-crf/blob/8f3203a1f1d7984c87718bfe31853242670258db/torchcrf/__init__.py#L259

    Args:
        emissions `torch.FloatTensor`: (batch_size, seq_length, num_classes)

        start_transitions `torch.FloatTensor`: (num_classes, )

        end_transitions `torch.FloatTensor`: (num_classes, )

        transitions `torch.FloatTensor`: (num_classes, num_classes)

        masks `torch.BoolTensor`: (seq_length, batch_size)
            optional, default=None

    Returns:
        sequences `torch.LongTensor`: (batch_size, seq_length)
    """
    batch_size, seq_length = emissions.shape[:2]

    if masks is None:
        masks = torch.ones(
            (batch_size, seq_length), dtype=torch.bool, device=emissions.device
        )

    emissions = emissions.transpose(0, 1)
    masks = masks.transpose(0, 1)

    with torch.no_grad():
        # Start transition and first emission
        # shape: (batch_size, num_classes)
        score = start_transitions + emissions[0]
        history = []

        # score is a tensor of size (batch_size, num_classes)
        #   where for every batch, value at column j stores
        #   the score of the best tag sequence so far that ends
        #   with tag j history saves where the best tags
        #   candidate transitioned from; this is used
        #   when we trace back the best tag sequence

        # Viterbi algorithm recursive case:
        #   we compute the score of the best tag sequence
        #   for every possible next tag
        for i in range(1, seq_length):
            # Broadcast viterbi score for every possible next tag
            # shape: (batch_size, num_classes, 1)
            broadcast_score = score.unsqueeze(2)

            # Broadcast emission score for every possible current tag
            # shape: (batch_size, 1, num_classes)
            broadcast_emission = emissions[i].unsqueeze(1)

            # Compute the score tensor of size
            #   (batch_size, num_classes, num_classes)
            #   where for each sample, entry at row i and column j
            #   stores the score of the best tag sequence so far
            #   that ends with transitioning from tag i to tag j and emitting
            # shape: (batch_size, num_classes, num_classes)
            next_score = broadcast_score + transitions + broadcast_emission

            # Find the maximum score over all possible current tag
            # shape: (batch_size, num_classes)
            next_score, indices = next_score.max(dim=1)

            # Set score to the next score if this timestep is valid
            #   (masks == 1) and save the index that produces the next score
            # shape: (batch_size, num_classes)
            score = torch.where(masks[i].unsqueeze(1), next_score, score)
            history.append(indices)

        # End transition score
        # shape: (batch_size, num_classes)
        score += end_transitions

        # Now, compute the best path for each sample

        # shape: (batch_size,)
        seq_ends = masks.long().sum(dim=0) - 1
        seqs = []

        for idx in range(batch_size):
            # Find the tag which maximizes the score at the last timestep;
            #   this is our best tag for the last timestep
            _, best_last_tag = score[idx].max(dim=0)
            best_tags = [best_last_tag.item()]

            # We trace back where the best last tag comes from,
            #   append that to our best tag sequence,
            #   and trace it back again, and so on
            for hist in reversed(history[: seq_ends[idx]]):
                best_last_tag = hist[idx][best_tags[-1]]
                best_tags.append(best_last_tag.item())

            # Reverse the order because we start from the last timestep
            best_tags.reverse()

            # padding individual sequence on the fly
            # such padding method is used because to generate sequences
            #   each of length=`seq_length` even if width of boolean masks for
            #   entire batch is lesser than `seq_length`
            seqs.append(
                torch.nn.functional.pad(
                    input=torch.tensor(
                        best_tags, dtype=torch.long, device=emissions.device
                    ),
                    pad=(0, seq_length - len(best_tags)),
                    mode="constant",
                    value=Constants.PAD_TOKEN_ENTITY_INT_LABEL.value,
                )
            )

        seqs = torch.stack(tensors=seqs)
    return seqs
