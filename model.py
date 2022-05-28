import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

import random
import math
import time
from tqdm import tqdm

import perception
from utils import INP_VOCAB, DEVICE, START, NULL, END
from rnn import RNNModel

class EmbeddingIn(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.image_input = config.input == 'image'

        if self.image_input:
            self.image_encoder = perception.make_model(config)
        self.n_token = config.in_vocab_size
        self.embedding = nn.Embedding(self.n_token, config.emb_dim)
        if config.embedding_init == "xavier":
            torch.nn.init.xavier_uniform_(self.embedding.weight)
        elif config.embedding_init == "kaiming":
            torch.nn.init.kaiming_normal_(self.embedding.weight)
        
    def forward(self, src, src_len):
        if self.image_input:
            src = self.image_encoder(src)
        else:
            src = self.embedding(src)

        max_len = src_len.max()
        current = 0
        padded_src = []
        emb_start = self.embedding(torch.tensor([self.n_token - 3]).to(src.device))
        emb_end = self.embedding(torch.tensor([self.n_token - 2]).to(src.device))
        emb_null = self.embedding(torch.tensor([self.n_token - 1]).to(src.device))
        for l in src_len:
            current_input = src[current:current+l]
            current_input = [emb_start, current_input, emb_end] + [emb_null] * (max_len - l) 
            current_input = torch.cat(current_input)
            padded_src.append(current_input)
            current += l
        src = torch.stack(padded_src)
        src_len = src_len + 2 # we append a start and end to every seq
        return src, src_len


class NeuralArithmetic(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        config.in_vocab_size = len(INP_VOCAB)
        config.out_vocab_size = len(config.res_enc.vocab)
        config.decoder_sos = config.res_enc.start_idx
        config.decoder_eos = config.res_enc.end_idx

        config.embedding_init = 'pytorch'
        if 'scaledinit' in config.model:
            config.embedding_init = 'kaiming'
        elif 'opennmt' in config.model:
            config.embedding_init = 'xavier'

        self.embedding_in = EmbeddingIn(config)
        if config.model.startswith('TRAN.'):
            import transformer
            self.model = transformer.create_model(config)
        else:
            self.model = RNNModel(config)
    
    def forward(self, src, tgt, src_len):
        src, src_len = self.embedding_in(src, src_len)
        output = self.model(src, src_len, tgt)
        return output

def make_model(config):
    model = NeuralArithmetic(config)
    return model