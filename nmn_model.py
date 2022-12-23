import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

import random
import math
import time
from tqdm import tqdm
from copy import deepcopy

import perception
from utils import INP_VOCAB, SYM2PROG, ID2SYM, SYM2ID, NULL_VALUE
from rnn import RNNModel
from model import EmbeddingIn

class GTExecutionEngine:
	def __init__(self):
		self.id2fn = {}
		for s, f in SYM2PROG.items():
			i = SYM2ID(s)
			self.id2fn[i] = f

	def single_call(self, program):
		# program: postfix expression, a.k.a., Reverse Polish Notation 
		# https://leetcode.com/problems/evaluate-reverse-polish-notation/

		stack = []
		for i in program:
			fn = self.id2fn.get(i, None)
			if not fn or len(stack) < fn.arity:
				return NULL_VALUE
			out = fn(*stack[-fn.arity:])
			if out is NULL_VALUE:
				return NULL_VALUE
			del stack[-fn.arity:]
			stack.append(out)

		if len(stack) == 1:
			return stack[0]
		else:
			return NULL_VALUE

	def __call__(self, programs):
		outputs = [self.single_call(pg) for pg in programs]
		return outputs


class NeuralModuleNetwork(nn.Module):
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

        pg_config = deepcopy(config)
        pg_config.out_vocab_size = len(INP_VOCAB)
        self.program_generator = RNNModel(pg_config)
        self.execution_engine = GTExecutionEngine()

    def forward(self, src, programs, tgt, src_len):
        src, src_len = self.embedding_in(src, src_len)
        programs_pred = self.program_generator(src, src_len, programs, teacher_forcing_ratio=0.)
        results_pred = self.execution_engine(programs_pred)
        return programs_pred, results_pred

def make_model(config):
    model = NeuralModuleNetwork(config)
    return model