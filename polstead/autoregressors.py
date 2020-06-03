""" Wrappers around autoregressive torch modules. """
import torch
import torch.nn as nn

from asta import Tensor, dims

SEQ_LEN = dims.SEQ_LEN
IN_SIZE = dims.IN_SIZE
HIDDEN_SIZE = dims.HIDDEN_SIZE
OUT_SIZE = dims.OUT_SIZE
BATCH_SIZE = dims.BATCH_SIZE
NUM_DIRS = dims.NUM_DIRS


class LSTM(nn.Module):
    """ A simple LSTM module. """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int,
        dropout: float,
        bi: bool,
    ):
        super().__init__()
        num_dirs = 2 if bi else 1
        batch_size = 1
        self.batch_size = batch_size
        self.output_size = output_size
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bi,
        )
        self.linear = nn.Linear(num_dirs * hidden_size, output_size)

        # LSTM states.
        self.hidden = torch.zeros((num_layers * num_dirs, batch_size, hidden_size))
        self.cell = torch.zeros((num_layers * num_dirs, batch_size, hidden_size))

    def forward(
        self, x: Tensor[float, (SEQ_LEN, IN_SIZE)]
    ) -> Tensor[float, (SEQ_LEN, OUT_SIZE)]:
        """ Execute a forward pass through the module. """
        self.hidden.detach_()
        self.cell.detach_()

        logits: Tensor[float, (BATCH_SIZE, SEQ_LEN, NUM_DIRS * HIDDEN_SIZE)]

        # Pass the input through the LSTM layers.
        x = x.unsqueeze(0)
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        logits, states = self.lstm(x, (self.hidden, self.cell))

        # Update the LSTM states.
        self.hidden, self.cell = states

        # Pass the LSTM last layer logits/outs through linear layer.
        outs: Tensor[float, (BATCH_SIZE, SEQ_LEN, OUT_SIZE)] = self.linear(logits)

        return outs[0]
