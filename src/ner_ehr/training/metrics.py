import numpy as np

import torch
from sklearn.metrics import confusion_matrix
from torch import nn
from torch import Tensor


def accuracy_per_class(Y_hat: Tensor, Y: Tensor):
    """
    Args:
        Y_hat: (B, S, num_classes)

        Y: (B, S)
            B: Batch size

            S: seq_length

            num_classes: number of classes/labels

    Returns:
        A array with accuracies per class label
    """

    X1 = Y.view(-1)
    if "cuda" in X1.device.type:
        X1 = X1.cpu()

    with torch.no_grad():
        X2 = torch.argmax(nn.functional.softmax(Y_hat, dim=-1), dim=-1).view(
            -1
        )
        if "cuda" in X2.device.type:
            X2 = X2.cpu()
        labels = np.arange(Y_hat.size(-1))

    return confusion_matrix(
        y_true=X1, y_pred=X2, labels=labels, normalize="true"
    ).diagonal()
