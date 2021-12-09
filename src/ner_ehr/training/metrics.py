from typing import Optional, Tuple

import numpy as np
import torch
from sklearn import metrics

from ner_ehr.decode import argmax_decode, viterbi_decode

DECODE_MODE: str = "argmax"
START_TRANSITIONS: torch.FloatTensor = None
END_TRANSITIONS: torch.FloatTensor = None
TRANSITIONS: torch.FloatTensor = None
MASKS: Optional[torch.BoolTensor] = None


def _prepare_true_pred(
    Y_hat: torch.FloatTensor,
    Y: torch.LongTensor,
    decode_mode: str = DECODE_MODE,
    start_transitions: torch.FloatTensor = START_TRANSITIONS,
    end_transitions: torch.FloatTensor = END_TRANSITIONS,
    transitions: torch.FloatTensor = TRANSITIONS,
    masks: Optional[torch.BoolTensor] = MASKS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Helper function to decode predicted sequence as per given method."""
    if decode_mode == "argmax":
        pred = argmax_decode(emissions=Y_hat)
    else:
        pred = viterbi_decode(
            emissions=Y_hat,
            start_transitions=start_transitions,
            end_transitions=end_transitions,
            transitions=transitions,
            masks=masks,
        )

    with torch.no_grad():
        pred = pred.view(-1)
        if "cuda" in pred.device.type:
            pred = pred.cpu()

        true = Y.view(-1)
        if "cuda" in true.device.type:
            true = true.cpu()

        labels = np.arange(Y_hat.size(-1))

    return true.numpy(), pred.numpy(), labels


def accuracy_score(
    Y_hat: torch.FloatTensor,
    Y: torch.LongTensor,
    decode_mode: str = DECODE_MODE,
    start_transitions: torch.FloatTensor = START_TRANSITIONS,
    end_transitions: torch.FloatTensor = END_TRANSITIONS,
    transitions: torch.FloatTensor = TRANSITIONS,
    masks: Optional[torch.BoolTensor] = MASKS,
) -> Tuple[float, np.ndarray]:
    """
    Args:
        Y_hat: (B, S, num_classes)

        Y: (B, S)
            B: Batch size

            S: seq_length

            num_classes: number of classes/labels

    Returns:
        A Tuple with overall accuracy and array with accuracies per class label
    """
    y_true, y_pred, _ = _prepare_true_pred(
        Y_hat=Y_hat,
        Y=Y,
        decode_mode=decode_mode,
        start_transitions=start_transitions,
        end_transitions=end_transitions,
        transitions=transitions,
        masks=masks,
    )
    return metrics.accuracy_score(y_true=y_true, y_pred=y_pred)


def accuracy_scores(
    Y_hat: torch.FloatTensor,
    Y: torch.LongTensor,
    decode_mode: str = DECODE_MODE,
    start_transitions: torch.FloatTensor = START_TRANSITIONS,
    end_transitions: torch.FloatTensor = END_TRANSITIONS,
    transitions: torch.FloatTensor = TRANSITIONS,
    masks: Optional[torch.BoolTensor] = MASKS,
) -> Tuple[float, np.ndarray]:
    """
    Args:
        Y_hat: (B, S, num_classes)

        Y: (B, S)
            B: Batch size

            S: seq_length

            num_classes: number of classes/labels

    Returns:
        A Tuple with overall accuracy and array with accuracies per class label
    """
    y_true, y_pred, labels = _prepare_true_pred(
        Y_hat=Y_hat,
        Y=Y,
        decode_mode=decode_mode,
        start_transitions=start_transitions,
        end_transitions=end_transitions,
        transitions=transitions,
        masks=masks,
    )
    return metrics.confusion_matrix(
        y_true=y_true, y_pred=y_pred, labels=labels, normalize="true"
    ).diagonal()


def confusion_matrix(
    Y_hat: torch.FloatTensor,
    Y: torch.LongTensor,
    decode_mode: str = DECODE_MODE,
    start_transitions: torch.FloatTensor = START_TRANSITIONS,
    end_transitions: torch.FloatTensor = END_TRANSITIONS,
    transitions: torch.FloatTensor = TRANSITIONS,
    masks: Optional[torch.BoolTensor] = MASKS,
) -> Tuple[float, np.ndarray]:
    """
    Args:
        Y_hat: (B, S, num_classes)

        Y: (B, S)
            B: Batch size

            S: seq_length

            num_classes: number of classes/labels

    Returns:
        A Tuple with overall accuracy and array with accuracies per class label
    """
    y_true, y_pred, labels = _prepare_true_pred(
        Y_hat=Y_hat,
        Y=Y,
        decode_mode=decode_mode,
        start_transitions=start_transitions,
        end_transitions=end_transitions,
        transitions=transitions,
        masks=masks,
    )
    return metrics.confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        labels=labels,
    )


def all_metrics(
    Y_hat: torch.FloatTensor,
    Y: torch.LongTensor,
    decode_mode: str = DECODE_MODE,
    start_transitions: torch.FloatTensor = START_TRANSITIONS,
    end_transitions: torch.FloatTensor = END_TRANSITIONS,
    transitions: torch.FloatTensor = TRANSITIONS,
    masks: Optional[torch.BoolTensor] = MASKS,
    eps: float = 1e-64,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Args:
        Y_hat: (B, S, num_classes)

        Y: (B, S)
            B: Batch size

            S: seq_length

            num_classes: number of classes/labels

    Returns:
        A Tuple with overall accuracy and array with accuracies per class label
    """
    y_true, y_pred, labels = _prepare_true_pred(
        Y_hat=Y_hat,
        Y=Y,
        decode_mode=decode_mode,
        start_transitions=start_transitions,
        end_transitions=end_transitions,
        transitions=transitions,
        masks=masks,
    )
    acc = metrics.accuracy_score(y_true=y_true, y_pred=y_pred)
    cm = metrics.confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        labels=labels,
    )
    accs = (cm / (cm.sum(axis=1, keepdims=True) + eps)).diagonal()

    return acc, accs, cm
