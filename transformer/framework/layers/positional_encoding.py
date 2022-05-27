import torch
import torch.nn
import math
from typing import Optional
import numpy as np

def sinusoidal_pos_embedding(d_model: int, max_len: int = 5000, pos_offset: int = 0,
                             device: Optional[torch.device] = None, max_allowed=None):
    pe = torch.zeros(max_len, d_model, device=device)
    position = np.expand_dims(np.arange(0, max_len), 1) + pos_offset
    if max_allowed:
        cap_position = np.minimum(np.abs(position), max_allowed)
        cap_position[position < 0] *= -1
        position = cap_position
    position = torch.tensor(position, dtype=torch.float, device=device)
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float, device=device) * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class PositionalEncoding(torch.nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
        batch_first: if true, batch dimension is the first, if not, its the 2nd.
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000, batch_first: bool = False,
                 scale: float = 1, emb_type='sin'):
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        self.batch_dim = 0 if batch_first else 1
        if emb_type == 'sin':
            pe = sinusoidal_pos_embedding(d_model, max_len, 0) * scale
            pe = pe.unsqueeze(self.batch_dim)
            self.register_buffer('pe', pe)
        elif emb_type == 'learn':
            shape = (1, max_len, d_model) if batch_first else (max_len, 1, d_model)
            pe = torch.nn.Parameter(torch.Tensor(*shape))
            torch.nn.init.normal_(pe)
            self.pe = pe

    def get(self, n: int, offset: int) -> torch.Tensor:
        return self.pe.narrow(1 - self.batch_dim, start=offset, length=n)

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        x = x + self.get(x.size(1 - self.batch_dim), offset)
        return self.dropout(x)
