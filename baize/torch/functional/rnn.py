# coding: utf-8
# 2021/8/16 @ tongshiwei

import torch
from torch.nn import LSTM, LSTMCell, RNNBase, RNNCellBase

__all__ = ["get_rnn_state_shape", "get_rnn_init_state"]


def get_rnn_state_shape(rnn: (str, RNNBase, RNNCellBase), batch_size, num_layers=None, hidden_size=None):
    """

    Parameters
    ----------
    rnn
    batch_size
    num_layers
    hidden_size

    Returns
    -------

    Examples
    ---------
    >>> import torch
    >>> from torch import nn
    >>> rnn = nn.RNN(10, 20, 2)
    >>> input = torch.randn(5, 3, 10)
    >>> h0 = torch.randn(get_rnn_state_shape(rnn, input.shape[1]))
    >>> output, hn = rnn(input, h0)
    """
    if isinstance(rnn, str) is False:
        hidden_size: int = rnn.hidden_size
        num_layers: int = rnn.num_layers
        if rnn.bidirectional is True:
            num_layers *= 2

    return num_layers, batch_size, hidden_size


def get_rnn_init_state(rnn: (str, RNNBase, RNNCellBase), batch_size, num_layers=None, hidden_size=None):
    """

    Parameters
    ----------
    rnn: str, RNNBase, RNNCellBase

    num_layers
    batch_size
    hidden_size

    Returns
    -------

    Examples
    --------
    >>> h0, c0 = get_rnn_init_state("lstm", 16, 2, 3)
    >>> h0.shape
    torch.Size([2, 16, 3])
    >>> c0.shape
    torch.Size([2, 16, 3])
    >>> h0 = get_rnn_init_state("rnn",  16, 2, 3)
    >>> h0.shape
    torch.Size([2, 16, 3])
    >>> from torch import nn
    >>> rnn = nn.LSTM(4, 3, bidirectional=True)
    >>> h0, c0 = get_rnn_init_state(rnn, 16)
    >>> h0.shape
    torch.Size([2, 16, 3])
    >>> c0.shape
    torch.Size([2, 16, 3])
    >>> rnn = nn.LSTM(4, 3, 2, bidirectional=True)
    >>> h0, c0 = get_rnn_init_state(rnn, 16)
    >>> h0.shape
    torch.Size([4, 16, 3])
    >>> c0.shape
    torch.Size([4, 16, 3])
    >>> import torch
    >>> rnn = nn.LSTM(10, 20, 2)
    >>> input = torch.randn(5, 3, 10)
    >>> output, (hn, cn) = rnn(input, get_rnn_init_state(rnn, input.shape[1]))
    >>> output.shape
    torch.Size([5, 3, 20])
    >>> rnn = nn.RNN(10, 20, 2)
    >>> input = torch.randn(5, 3, 10)
    >>> output, hn = rnn(input, get_rnn_init_state(rnn, input.shape[1]))
    """

    num_layers, batch_size, hidden_size = get_rnn_state_shape(rnn, batch_size, num_layers, hidden_size)

    if isinstance(rnn, str) is False:
        if isinstance(rnn, (LSTM, LSTMCell)):
            rnn = "lstm"
        else:
            rnn = "rnn"

    h0 = torch.zeros(num_layers, batch_size, hidden_size)

    if rnn == "lstm":
        c0 = torch.zeros(num_layers, batch_size, hidden_size)
        return h0, c0
    else:
        return h0
