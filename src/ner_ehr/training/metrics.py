"""This module contains evaluation metrics."""
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
    """Decode predicted sequence as per given decode mode.

    Args:
        Y_hat: tensor of shape (batch_size, seq_length, num_classes)
            contains unnormalized scores across classes

        Y: tensor of shape (batch_size, num_classes)

        decode_mode: decoding mode, available modes: [`argmax`, `viterbi`],
            default=`argmax`

        start_transitions: tensor of shape (num_classes, )
            default=None, required when decode_mode=`viterbi`

        end_transitions: tensor of shape (num_classes, )
            default=None, required when decode_mode=`viterbi`

        transitions: tensor of shape (num_classes, num_classes)
            default=None, required when decode_mode=`viterbi`

        masks: tensor of shape (batch_size, seq_length)
            default=None, required when decode_mode=`viterbi`
            but can be optional for viterbi decoding

    Returns:
        A tuple of (true, predicted, labels)
            true: 1-D NumPy array of shape (batch_size*seq_length, )
                containing true/gold labels

            pred: 1-D NumPy array of shape (batch_size*seq_length, )
                containing true/gold labels

            labels: 1-D NumPy array of shape (num_classes, )
    """
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
) -> float:
    """Compute and return accuracy between Y_hat and Y using given decode mode.

    Args:
        Y_hat: tensor of shape (batch_size, seq_length, num_classes)
            contains unnormalized scores across classes

        Y: tensor of shape (batch_size, num_classes)

        decode_mode: decoding mode, available modes: [`argmax`, `viterbi`],
            default=`argmax`

        start_transitions: tensor of shape (num_classes, )
            default=None, required when decode_mode=`viterbi`

        end_transitions: tensor of shape (num_classes, )
            default=None, required when decode_mode=`viterbi`

        transitions: tensor of shape (num_classes, num_classes)
            default=None, required when decode_mode=`viterbi`

        masks: tensor of shape (batch_size, seq_length)
            default=None, required when decode_mode=`viterbi`
            but can be optional for viterbi decoding

    Returns:
        A overall scalar accuracy between Y_hat and Y
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
) -> np.ndarray:
    """Compute and return accuracies between Y_hat and Y across all classes
        separately as per given decoding mode.

    Args:
        Y_hat: tensor of shape (batch_size, seq_length, num_classes)
            Contains unnormalized scores across classes

        Y: tensor of shape (batch_size, num_classes)

        decode_mode: decoding mode, available modes: [`argmax`, `viterbi`],
            default=`argmax`

        start_transitions: tensor of shape (num_classes, )
            default=None, required when decode_mode=`viterbi`

        end_transitions: tensor of shape (num_classes, )
            default=None, required when decode_mode=`viterbi`

        transitions: tensor of shape (num_classes, num_classes)
            default=None, required when decode_mode=`viterbi`

        masks: tensor of shape (batch_size, seq_length)
            default=None, required when decode_mode=`viterbi`
            but can be optional for viterbi decoding

    Returns:
        A 1-D NumPy array of shape (num_classes, )
            with accuracies between Y_hat and Y across each class
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
    """Compute and return confusion matrix between Y_hat and Y
        as per given decoding mode.

    Args:
        Y_hat: tensor of shape (batch_size, seq_length, num_classes)
            contains unnormalized scores across classes

        Y: tensor of shape (batch_size, num_classes)

        decode_mode: decoding mode, available modes: [`argmax`, `viterbi`],
            default=`argmax`

        start_transitions: tensor of shape (num_classes, )
            default=None, required when decode_mode=`viterbi`

        end_transitions: tensor of shape (num_classes, )
            default=None, required when decode_mode=`viterbi`

        transitions: tensor of shape (num_classes, num_classes)
            default=None, required when decode_mode=`viterbi`

        masks: tensor of shape (batch_size, seq_length)
            default=None, required when decode_mode=`viterbi`
            but can be optional for viterbi decoding

    Returns:
        A 2-D NumPy array of shape (num_classes, num_classes)
            containing values of confusion matrix generated between
            Y_hat and Y
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
    """Compute and return overall accuracy, accuracy per class and
        confusion matrix between Y_hat and Y as per given decoding mode.

    Args:
        Y_hat: tensor of shape (batch_size, seq_length, num_classes)
            contains unnormalized scores across classes

        Y: tensor of shape (batch_size, num_classes)

        decode_mode: decoding mode, available modes: [`argmax`, `viterbi`],
            default=`argmax`

        start_transitions: tensor of shape (num_classes, )
            default=None, required when decode_mode=`viterbi`

        end_transitions: tensor of shape (num_classes, )
            default=None, required when decode_mode=`viterbi`

        transitions: tensor of shape (num_classes, num_classes)
            default=None, required when decode_mode=`viterbi`

        masks: tensor of shape (batch_size, seq_length)
            default=None, required when decode_mode=`viterbi`
            but can be optional for viterbi decoding

        eps: small float value used for numerical stability

    Returns:
        A tuple containing (accuracy, accuracies, confusion_matrix)
            accuracy: overall scalar accuracy between Y_hat and Y
            accuracies: 1-D NumPy array of shape (num_classes, )
                with accuracies between Y_hat and Y across each class
            confusion_matrix: A 2-D NumPy array of
                shape (num_classes, num_classes) containing values from
                confusion matrix generated between Y_hat and Y
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
