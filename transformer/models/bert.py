import torch
import torch.nn
import torch.nn.functional as F
import framework
from layers import Transformer, TiedEmbedding
from typing import Callable, Optional
import math
import random
from sin_decoder import SinDecoder

class BertModel(torch.nn.Module):
    def __init__(self, n_input_tokens: int, state_size: int = 512, ff_multiplier: float = 4,
                 max_len: int=5000, transformer = Transformer,
                 pos_embeddig: Optional[Callable[[torch.Tensor, int], torch.Tensor]] = None, pos_emb_type='sin',
                 embedding_init: str = "pytorch",
                 in_embedding_size: Optional[int] = None, out_embedding_size: Optional[int] = None, 
                 scale_mode: str = "none", **kwargs):
        '''
        BERT model.

        :param n_input_tokens: Number of channels for the input vectors
        :param state_size: The size of the internal state of the transformer
        '''
        super().__init__()

        assert scale_mode in ["none", "opennmt", "down"]
        assert embedding_init in ["pytorch", "xavier", "kaiming"]


        self.cls_idx = 0 # the idx of cls token
        self.state_size = state_size
        self.embedding_init = embedding_init
        self.ff_multiplier = ff_multiplier
        self.n_input_tokens = n_input_tokens
        self.in_embedding_size = in_embedding_size
        self.out_embedding_size = out_embedding_size
        self.scale_mode = scale_mode
        self.src_pos = pos_embeddig or framework.layers.PositionalEncoding(state_size, max_len=max_len, batch_first=True,
                                emb_type=pos_emb_type, scale=(1.0 / math.sqrt(state_size)) if scale_mode == "down" else 1.0)
        
        self.register_buffer('int_seq', torch.arange(max_len, dtype=torch.long))
        self.construct(transformer, **kwargs)

    def pos_embed(self, t: torch.Tensor, offset: int=0) -> torch.Tensor:
        if self.scale_mode == "opennmt":
            t = t * math.sqrt(t.shape[-1])
        return self.src_pos(t, offset)

    def construct(self, transformer, **kwargs):
        if self.in_embedding_size is not None:
            self.in_embedding_upscale = torch.nn.Linear(self.in_embedding_size, self.state_size)

        self.trafo = transformer(d_model=self.state_size, dim_feedforward=int(self.ff_multiplier*self.state_size),
                                 **kwargs)
        self.trafo.decoder = None
        self.result_decoder = SinDecoder(inp_dim=self.state_size, res_dim=self.out_embedding_size, 
                            feedforward_dims=[self.state_size], dropout=kwargs['dropout'])

    def generate_len_mask(self, max_len: int, len: torch.Tensor) -> torch.Tensor:
        return self.int_seq[: max_len] >= len.unsqueeze(-1)


    def input_embed(self, src: torch.Tensor) -> torch.Tensor:
        if self.in_embedding_size is not None:
            src = self.in_embedding_upscale(src)
        return src

    def forward(self, src, src_len, tgt): 
        '''
        Run transformer encoder-decoder on some input/output pair

        :param src: source tensor. Shape: [N, S, D], where S is the in sequence length, N is the batch size
        :param src_len: length of source sequences. Shape: [N], N is the batch size
        :return: prediction of the target tensor. Shape [N, T, C_out]
        '''

        src = self.pos_embed(self.input_embed(src))
        src_len = src_len.to(src.device)

        n_steps = src.shape[1]
        in_len_mask = self.generate_len_mask(n_steps, src_len)
        memory = self.trafo.encoder(src, mask=in_len_mask)
        output = self.result_decoder(memory[:, self.cls_idx])

        return output