import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

import random
import math
import time
from tqdm import tqdm

from on_lstm import ONLSTMStack
from ordered_memory import OrderedMemory


def create_padding_mask(lens):
	# 1, pos is masked and not allowed to attend;
	max_len = max(lens)
	batch_size = len(lens)
	mask = np.ones((batch_size, max_len+2)) # 2 for the START, END token
	for i, l in enumerate(lens):
		mask[i, :l+2] = 0 
	return mask.astype(bool)


class RNNModel(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.config = config

		emb_dim = config.emb_dim
		hid_dim = config.hid_dim
		enc_layers = config.enc_layers
		dec_layers = config.dec_layers
		dropout = config.dropout

		if config.seq2seq == 'LSTM':
			self.dec_hid_dim = hid_dim * 2
			self.encoder = nn.LSTM(emb_dim, hid_dim, enc_layers, dropout=dropout, bidirectional=True)
			self.decoder = nn.LSTM(emb_dim, self.dec_hid_dim, dec_layers, dropout=dropout, bidirectional=False)
		elif config.seq2seq == 'GRU':
			self.dec_hid_dim = hid_dim * 2
			self.encoder = nn.GRU(emb_dim, hid_dim, enc_layers, dropout=dropout, bidirectional=True)
			self.decoder = nn.GRU(emb_dim, self.dec_hid_dim, dec_layers, dropout=dropout, bidirectional=False)
		elif config.seq2seq == 'ON':
			self.dec_hid_dim = hid_dim
			self.encoder = ONLSTMStack([emb_dim] + [hid_dim] * enc_layers, chunk_size=8, dropconnect=dropout, dropout=dropout)
			self.decoder = nn.LSTM(emb_dim, self.dec_hid_dim, dec_layers, dropout=dropout, bidirectional=False)
		elif config.seq2seq == 'OM':
			self.dec_hid_dim = hid_dim
			self.encoder = OrderedMemory(emb_dim, hid_dim, nslot=21, bidirection=False)
			self.decoder = nn.GRU(emb_dim, self.dec_hid_dim, dec_layers, dropout=dropout, bidirectional=False)
		else:
			pass

		self.embedding_out = nn.Embedding(config.out_vocab_size, config.emb_dim)
		self.classifier_out = nn.Linear(self.dec_hid_dim, config.out_vocab_size)

	def forward(self, src, src_len, tgt, tgt_len, teacher_forcing_ratio=0.5):
		src = src.transpose(0, 1)
		tgt = tgt.transpose(0, 1)
		if self.config.seq2seq == 'GRU':
			_, hidden = self.encoder(src)
			hidden = hidden.view(-1, 2, *hidden.shape[1:])[-1].transpose(0, 1)
			hidden = hidden.contiguous().view(hidden.shape[0], -1)
			hidden = torch.stack([hidden] * self.config.dec_layers)
		elif self.config.seq2seq == 'LSTM':
			_, (h, c) = self.encoder(src)
			h = h.view(-1, 2, *h.shape[1:])[-1].transpose(0, 1)
			h = h.contiguous().view(h.shape[0], -1)
			h = torch.stack([h] * self.config.dec_layers)

			c = c.view(-1, 2, *c.shape[1:])[-1].transpose(0, 1)
			c = c.contiguous().view(c.shape[0], -1)
			c = torch.stack([c] * self.config.dec_layers)

			hidden = (h, c)

		elif self.config.seq2seq == 'ON':
			h, c = self.encoder(src)[1][-1]
			h = torch.stack([h] * self.config.dec_layers)

			c = c.contiguous().view(c.shape[0], -1)
			c = torch.stack([c] * self.config.dec_layers)

			hidden = (h, c)

		elif self.config.seq2seq == 'OM':
			mask = torch.from_numpy(create_padding_mask(src_len)).to(src.device).transpose(0,1).contiguous()
			h = self.encoder(src, mask, output_last=True)
			h = torch.stack([h] * self.config.dec_layers)
			hidden = h

		use_teacher_forcing = True if self.training and random.random() < teacher_forcing_ratio else False
		if use_teacher_forcing:
			tgt = self.embedding_out(tgt)
			output, _ = self.decoder(tgt, hidden)
			output = self.classifier_out(F.relu(output))
		else:
			decoder_input = torch.tensor([[self.config.decoder_sos] * src.size(1)], device=src.device)
			output_list = []

			for i in range(tgt.size(0)):
				decoder_input = self.embedding_out(decoder_input)
				output, hidden = self.decoder(decoder_input, hidden)
				output = output[0]
				output = self.classifier_out(F.relu(output))
				topv, topi = output.topk(1)
				decoder_input = topi.view(1, -1).detach()
				output_list.append(output)
			
			output = torch.stack(output_list)
		return output.transpose(0, 1)
