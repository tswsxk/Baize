# coding: utf-8
# 2021/8/19 @ tongshiwei

import torch
from torch import Tensor

__all__ = ["tensor2list", "batch_select"]


def tensor2list(tensor: Tensor):
    """
    Convert a tensor into a list.

    Parameters
    ----------
    tensor: Tensor

    Returns
    -------
    obj: list

    Examples
    ---------
    >>> import torch
    >>> tensor = torch.ones(3)
    >>> tensor
    tensor([1., 1., 1.])
    >>> tensor2list(tensor)
    [1.0, 1.0, 1.0]
    """
    return tensor.cpu().tolist()


def batch_select(tensor: Tensor, batch_index: Tensor, keep_dim=False):
    """
    Select the row specified by the batch_index.

    Parameters
    ----------
    tensor: Tensor
        3-d tensor: (B, N, C)
    batch_index: Tensor
        1-d tensor: (B, )
    keep_dim: bool
        whether to keep the `N` (i.e., the second) dimension

    Returns
    -------
    tensor:

    Examples
    --------
    >>> tensor = torch.tensor([[[0, 1, 2], [1, 11, 12]], [[2, 21, 22], [3, 31, 32]]])
    >>> index = torch.tensor([0, 1])
    >>> batch_select(tensor, index)
    tensor([[ 0,  1,  2],
            [ 3, 31, 32]])
    >>> batch_select(tensor, index, keep_dim=True)
    tensor([[[ 0,  1,  2]],
    <BLANKLINE>
            [[ 3, 31, 32]]])
    """
    batch_index = batch_index.reshape(batch_index.shape[0], 1, 1)
    batch_index = batch_index.repeat(1, 1, tensor.shape[-1])
    tensor = torch.gather(tensor, 1, batch_index)
    if keep_dim is False:
        return tensor.squeeze(1)
    return tensor
