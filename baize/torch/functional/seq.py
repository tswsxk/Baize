# coding: utf-8
# 2021/8/19 @ tongshiwei
__all__ = ["pick", "length2mask", "get_sequence_mask", "mask_sequence"]

import torch
from torch import Tensor
from baize.torch.functional.utils import tensor2list


# from .utils import tensor2list


def pick(tensor: Tensor, index: Tensor, axis=-1):
    """
    Pick the elements specified by index


    Parameters
    ----------
    tensor: Tensor
        N-d tensor
    index: Tensor
        (N-1)-d tensor
    axis

    Returns
    -------
    pick_tensor: Tensor
        (N-1)-d tensor

    Examples
    --------
    >>> import torch
    >>> tensor = torch.tensor([[[0, 1], [10, 11], [20, 21]], [[30, 31], [40, 41], [50, 51]]])
    >>> tensor
    tensor([[[ 0,  1],
             [10, 11],
             [20, 21]],
    <BLANKLINE>
            [[30, 31],
             [40, 41],
             [50, 51]]])
    >>> index = torch.tensor([[0, 1, 0], [1, 1, 0]])
    >>> pick_tensor = pick(tensor, index)
    >>> pick_tensor
    tensor([[ 0, 11, 20],
            [31, 41, 50]])
    >>> tensor.shape, index.shape, pick_tensor.shape
    (torch.Size([2, 3, 2]), torch.Size([2, 3]), torch.Size([2, 3]))
    """
    return torch.gather(tensor, axis, index.unsqueeze(axis)).squeeze(axis)


def length2mask(length: (list, Tensor), max_len: int, valid_mask_val: ... = 1, invalid_mask_val: ... = 0, shape=None):
    """
    Generate `valid_mask_val`-`invalid_mask_val` full mask tensor based on the length tensor

    Parameters
    ----------
    length: list or Tensor
    max_len: int
    shape
    valid_mask_val:
        valid mask value
    invalid_mask_val:
        invalid mask value
    shape

    Returns
    -------
    mask: Tensor

    Examples
    --------
    >>> import torch
    >>> tensor = torch.ones(3, 4)
    >>> length = torch.tensor([1, 2, 3])
    >>> mask = length2mask(length, 3, [1, 1, 1, 1], [0, 0, 0, 0])
    >>> mask
    tensor([[[1, 1, 1, 1],
             [0, 0, 0, 0],
             [0, 0, 0, 0]],
    <BLANKLINE>
            [[1, 1, 1, 1],
             [1, 1, 1, 1],
             [0, 0, 0, 0]],
    <BLANKLINE>
            [[1, 1, 1, 1],
             [1, 1, 1, 1],
             [1, 1, 1, 1]]])
    >>> mask * tensor
    tensor([[[1., 1., 1., 1.],
             [0., 0., 0., 0.],
             [0., 0., 0., 0.]],
    <BLANKLINE>
            [[1., 1., 1., 1.],
             [1., 1., 1., 1.],
             [0., 0., 0., 0.]],
    <BLANKLINE>
            [[1., 1., 1., 1.],
             [1., 1., 1., 1.],
             [1., 1., 1., 1.]]])
    >>> length2mask(length, 3, shape=(4, ))
    tensor([[[1, 1, 1, 1],
             [0, 0, 0, 0],
             [0, 0, 0, 0]],
    <BLANKLINE>
            [[1, 1, 1, 1],
             [1, 1, 1, 1],
             [0, 0, 0, 0]],
    <BLANKLINE>
            [[1, 1, 1, 1],
             [1, 1, 1, 1],
             [1, 1, 1, 1]]])
    """
    mask = []

    if shape is not None:
        valid_mask_val = torch.full(shape, valid_mask_val)
        invalid_mask_val = torch.full(shape, invalid_mask_val)

    if isinstance(valid_mask_val, Tensor):
        valid_mask_val = tensor2list(valid_mask_val)
    if isinstance(invalid_mask_val, Tensor):
        invalid_mask_val = tensor2list(invalid_mask_val)
    if isinstance(length, Tensor):
        length = tensor2list(length)

    for _len in length:
        mask.append([valid_mask_val] * _len + [invalid_mask_val] * (max_len - _len))

    return torch.tensor(mask)


def get_sequence_mask(shape, sequence_length, axis=1):
    """
    Get the mask based on the sequence tensor shape and sequence_length

    Parameters
    ----------
    shape
    sequence_length
    axis

    Returns
    -------
    >>> import torch
    >>> seq = torch.ones(2, 4, 3)  # batch first
    >>> get_sequence_mask(seq, [2, 4])
    tensor([[[1., 1., 1.],
             [1., 1., 1.],
             [0., 0., 0.],
             [0., 0., 0.]],
    <BLANKLINE>
            [[1., 1., 1.],
             [1., 1., 1.],
             [1., 1., 1.],
             [1., 1., 1.]]])
    """
    shape = shape if not isinstance(shape, Tensor) else shape.shape
    assert axis <= len(shape)
    mask_shape = shape[axis + 1:]

    valid_mask_val = torch.ones(mask_shape)
    invalid_mask_val = torch.zeros(mask_shape)

    max_len = shape[axis]

    return length2mask(sequence_length, max_len, valid_mask_val, invalid_mask_val)


def mask_sequence(tensor: Tensor, sequence_length, axis=1):
    """
    Get the masked sequence based on the given sequence length


    Parameters
    ----------
    tensor
    sequence_length
    axis
        time axis

    Returns
    -------
    >>> import torch
    >>> seq = torch.ones(2, 4, 3)  # batch first
    >>> mask_sequence(seq, [2, 4])
    tensor([[[1., 1., 1.],
             [1., 1., 1.],
             [0., 0., 0.],
             [0., 0., 0.]],
    <BLANKLINE>
            [[1., 1., 1.],
             [1., 1., 1.],
             [1., 1., 1.],
             [1., 1., 1.]]])
    """
    mask = get_sequence_mask(tensor.shape, sequence_length, axis).to(tensor.device)
    return tensor * mask
