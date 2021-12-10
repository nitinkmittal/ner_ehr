from torch import nn
import torch


def cross_entropy(
    Y_hat: torch.FloatTensor, Y: torch.LongTensor, **kwargs
) -> torch.FloatTensor:
    """
    Compute cross-entropy loss between input (Y_hat) and target (Y) tensors.

    Args:
        Y_hat (FloatTensor): (B, S, num_classes)
            Values in Y_hat are expected to unnormalized scores

        Y (IntTensor or LongTensor): (B, S)
            Value in Y are expected to be actual class indices

        **kwargs: keyword arguments other than `input` and `target`
            for torch.nn.functional.cross_entropy
            https://github.com/pytorch/pytorch/blob/master/torch/nn/functional.py

    Returns:
        Scalar cross-entropy loss between Y_hat and Y
    """
    Y_hat = Y_hat.view(
        -1, Y_hat.shape[-1]
    )  # (B, S, num_classes) -> (B * S, num_classes)
    Y = Y.view(-1)  # (B, S) -> (B * S, )
    return nn.functional.cross_entropy(input=Y_hat, target=Y, **kwargs)
