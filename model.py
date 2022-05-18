import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

import random
import math
import time
from tqdm import tqdm

import sys
import resnet_scan
from utils import SYMBOLS, INP_VOCAB, RES_VOCAB, DEVICE, START, NULL, END, RES_MAX_LEN
from rnn import RNNModel

class EmbeddingIn(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.image_input = config.input == 'image'

        if self.image_input:
            self.image_encoder = resnet_scan.make_model(n_class=config.in_vocab_size - 3)
        self.n_token = config.in_vocab_size
        self.embedding = nn.Embedding(self.n_token, config.emb_dim)
        if config.embedding_init == "xavier":
            torch.nn.init.xavier_uniform_(self.embedding.weight)
        elif config.embedding_init == "kaiming":
            torch.nn.init.kaiming_normal_(self.embedding.weight)
        
    def forward(self, src, src_len):
        if self.image_input:
            logits = self.image_encoder(src)
            probs = F.softmax(logits, dim=-1)
            src = torch.matmul(probs, self.embedding.weight[:-3])
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
        return src


class NeuralArithmetic(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        config.in_vocab_size = len(INP_VOCAB)
        config.out_vocab_size = len(RES_VOCAB)
        config.decoder_sos = RES_VOCAB.index(START)
        config.decoder_eos = RES_VOCAB.index(END)

        config.embedding_init = 'pytorch'
        if config.seq2seq == 'transformer':
            if 'scaledinit' in config.transformer:
                config.embedding_init = 'kaiming'
            elif 'opennmt' in config.transformer:
                config.embedding_init = 'xavier'

        self.embedding_in = EmbeddingIn(config)
        if config.seq2seq in ['GRU', 'LSTM', 'ON', 'OM']:
            self.seq2seq = RNNModel(config)
        else:
            import transformer
            self.seq2seq = transformer.create_model(config)
    
    def forward(self, src, tgt, src_len, tgt_len):
        src = self.embedding_in(src, src_len)
        output = self.seq2seq(src, src_len, tgt, tgt_len)
        return output

def make_model(config):
    model = NeuralArithmetic(config)
    return model