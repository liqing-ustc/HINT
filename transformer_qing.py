import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

import random
import math
import time
from tqdm import tqdm

from utils import SYMBOLS, INP_VOCAB, RES_VOCAB, DEVICE, START, NULL, END, RES_MAX_LEN

class SinePositionalEncoding(nn.Module):
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
	Examples:
		>>> pos_encoder = PositionalEncoding(d_model)
	"""

	def __init__(self, d_model, dropout=0.1, max_len=5000):
		super(SinePositionalEncoding, self).__init__()
		self.dropout = nn.Dropout(p=dropout)

		pe = torch.zeros(max_len, d_model)
		position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
		pe[:, 0::2] = torch.sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term)
		pe = pe.unsqueeze(0).transpose(0, 1)
		self.register_buffer('pe', pe)

	def forward(self, x):
		r"""Inputs of forward function
		Args:
			x: the sequence fed to the positional encoder model (required).
		Shape:
			x: [sequence length, batch size, embed dim]
			output: [sequence length, batch size, embed dim]
		Examples:
			>>> output = pos_encoder(x)
		"""

		x = x + self.pe[:x.size(0), :]
		return self.dropout(x)

class LearnedPositionalEncoding(nn.Module):
	def __init__(self, d_model, dropout=0.1, max_len=100):
		super(LearnedPositionalEncoding, self).__init__()
		self.dropout = nn.Dropout(p=dropout)
		self.pe = nn.Embedding(max_len, d_model)

	def forward(self, x):
		r"""Inputs of forward function
		Args:
			x: the sequence fed to the positional encoder model (required).
		Shape:
			x: [sequence length, batch size, embed dim]
			output: [sequence length, batch size, embed dim]
		"""

		seq_len, batch_size = x.shape[:2]
		pos = torch.arange(0, seq_len).unsqueeze(1).to(DEVICE)
		x = x + self.pe(pos)
		return self.dropout(x)

def create_padding_mask(lens):
	# 1, pos is masked and not allowed to attend;
	max_len = max(lens)
	batch_size = len(lens)
	mask = np.ones((batch_size, max_len+2)) # 2 for the START, END token
	for i, l in enumerate(lens):
		mask[i, :l+2] = 0 
	return mask.astype(bool)

def create_padding_mask_tgt(tgt):
	mask = tgt == RES_VOCAB.index(NULL)
	return mask

class TransformerModel(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.config = config

		nhead = config.nhead
		emb_dim = config.emb_dim
		hid_dim = config.hid_dim
		enc_layers = config.enc_layers
		dec_layers = config.dec_layers
		dropout = config.dropout

		self.d_model = emb_dim
		# self.enc_pos_embedding = SinePositionalEncoding(emb_dim, dropout)
		# self.dec_pos_embedding = SinePositionalEncoding(emb_dim, dropout)
		self.enc_pos_embedding = LearnedPositionalEncoding(emb_dim, dropout)
		self.dec_pos_embedding = LearnedPositionalEncoding(emb_dim, dropout)
		self.transformer = nn.Transformer(emb_dim, nhead, enc_layers, dec_layers, hid_dim, dropout)

		self.embedding_out = nn.Embedding(len(RES_VOCAB), emb_dim)
		self.classifier_out = nn.Linear(emb_dim, len(RES_VOCAB))


	def forward(self, src, src_len, tgt, tgt_len):
		src = src.transpose(0, 1)
		tgt = tgt.transpose(0, 1)
		src_padding_mask = torch.from_numpy(create_padding_mask(src_len)).to(DEVICE)
		src = self.enc_pos_embedding(src * math.sqrt(self.d_model))
		if self.training:
			tgt_padding_mask = create_padding_mask_tgt(tgt).transpose(0, 1)
			tgt_mask = self.transformer.generate_square_subsequent_mask(len(tgt)).to(DEVICE)
			tgt_emb = self.embedding_out(tgt)
			tgt_emb = self.dec_pos_embedding(tgt_emb * math.sqrt(self.d_model))
			output = self.transformer(src, tgt_emb, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_padding_mask,
							src_key_padding_mask=src_padding_mask, memory_key_padding_mask=src_padding_mask)
			output = self.classifier_out(F.relu(output))
		else:
			tgt = tgt[:1]
			output_list = []
			finish = torch.zeros((src.shape[1])).bool().to(DEVICE)
			while not finish.all() and len(tgt) <= RES_MAX_LEN:
				tgt_padding_mask = create_padding_mask_tgt(tgt).transpose(0, 1)
				tgt_mask = self.transformer.generate_square_subsequent_mask(len(tgt)).to(DEVICE)
				tgt_emb = self.embedding_out(tgt)
				tgt_emb = self.dec_pos_embedding(tgt_emb * math.sqrt(self.d_model))
				output = self.transformer(src, tgt_emb, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_padding_mask,
							src_key_padding_mask=src_padding_mask, memory_key_padding_mask=src_padding_mask)
				output = output[-1]
				output = self.classifier_out(F.relu(output))
				pred = output.argmax(1)
				pred[finish] = RES_VOCAB.index(NULL)
				tgt = torch.cat([tgt, pred.unsqueeze(0)])
				finish[pred == RES_VOCAB.index(END)] = True

				output_list.append(output)
			
			output = torch.stack(output_list)
		return output.transpose(0, 1)