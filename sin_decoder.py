
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def sinusoidal_embedding(d_model, max_len=10000, device=None):
    emb = torch.zeros(max_len, d_model, device=device)
    position = torch.arange(0, max_len, dtype=torch.float, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float, device=device) * (-math.log(10000.0) / d_model))
    emb[:, 0::2] = torch.sin(position * div_term)
    emb[:, 1::2] = torch.cos(position * div_term)
    return emb

class SinDecoder(nn.Module):
	def __init__(self, inp_dim, res_dim, feedforward_dims=[], dropout=0.1):
		super().__init__()

		self.ffw = self.build_ffw([inp_dim] + feedforward_dims + [res_dim], dropout)

		res_emb = sinusoidal_embedding(res_dim)
		# res_emb = F.normalize(res_emb, dim=1)
		self.register_buffer('res_emb', res_emb)

	def build_ffw(self, dims, dropout):
		layers = []
		last_h = dims[0]
		for h in dims[1:-1]:
			layers.append(nn.Linear(last_h, h))
			layers.append(nn.ReLU())
			layers.append(nn.Dropout(dropout))
			last_h = h
		layers.append(nn.Linear(last_h, dims[-1]))
		return nn.Sequential(*layers)


	def forward(self, input):
		x = self.ffw(input)
		# x = torch.matmul(F.normalize(x, dim=1), self.res_emb.t())
		x = torch.matmul(x, self.res_emb.t())
		# x = x.unsqueeze(1) - self.res_emb.unsqueeze(0)
		# x = -(x**2).mean(axis=-1)
		return x




