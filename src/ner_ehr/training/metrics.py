from typing import Tuple

import numpy as np
import torch
from sklearn import metrics
from torch import Tensor, nn


def _prepare_true_pred(
    Y_hat: Tensor, Y: Tensor
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    true = Y.view(-1)
    if "cuda" in true.device.type:
        true = true.cpu()

    with torch.no_grad():
        pred = torch.argmax(nn.functional.softmax(Y_hat, dim=-1), dim=-1).view(
            -1
        )
        if "cuda" in pred.device.type:
            pred = pred.cpu()
        labels = np.arange(Y_hat.size(-1))
    return true, pred, labels


def accuracy_score(Y_hat: Tensor, Y: Tensor) -> Tuple[float, np.ndarray]:
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
    y_true, y_pred, _ = _prepare_true_pred(Y_hat=Y_hat, Y=Y)
    return metrics.accuracy_score(y_true=y_true, y_pred=y_pred)


def accuracy_scores(Y_hat: Tensor, Y: Tensor) -> Tuple[float, np.ndarray]:
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
    y_true, y_pred, labels = _prepare_true_pred(Y_hat=Y_hat, Y=Y)
    return metrics.confusion_matrix(
        y_true=y_true, y_pred=y_pred, labels=labels, normalize="true"
    ).diagonal()


def confusion_matrix(Y_hat: Tensor, Y: Tensor) -> Tuple[float, np.ndarray]:
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
    y_true, y_pred, labels = _prepare_true_pred(Y_hat=Y_hat, Y=Y)
    return metrics.confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        labels=labels,
    )


def all_metrics(
    Y_hat: Tensor, Y: Tensor, eps: float = 1e-64
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
    y_true, y_pred, labels = _prepare_true_pred(Y_hat=Y_hat, Y=Y)
    acc = metrics.accuracy_score(y_true=y_true, y_pred=y_pred)
    cm = metrics.confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        labels=labels,
    )
    accs = (cm / (cm.sum(axis=1, keepdims=True) + eps)).diagonal()

    return acc, accs, cm
