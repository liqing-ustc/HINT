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
from sin_decoder import SinDecoder


def create_padding_mask(lens):
    # 1, pos is masked and not allowed to attend;
    max_len = max(lens)
    batch_size = len(lens)
    mask = np.ones((batch_size, max_len+2)) # 2 for the START, END token
    for i, l in enumerate(lens):
        mask[i, :l+2] = 0 
    return mask.astype(bool)


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        
        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias = False)
        
    def forward(self, hidden, encoder_outputs):
        
        #hidden = [batch size, dec hid dim]
        #encoder_outputs = [src len, batch size, enc hid dim * 2]
        
        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]
        
        #repeat decoder hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        
        #hidden = [batch size, src len, dec hid dim]
        #encoder_outputs = [batch size, src len, enc hid dim * 2]
        
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim = 2))) 
        
        #energy = [batch size, src len, dec hid dim]

        attention = self.v(energy).squeeze(2)
        
        #attention= [batch size, src len]
        
        return F.softmax(attention, dim=1)

class Decoder(nn.Module):
    def __init__(self, rnn, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = rnn
        self.fc_out = nn.Linear(dec_hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input, hidden, encoder_outputs=None):
        embedded = self.dropout(self.embedding(input))
        output, hidden = self.rnn(embedded, hidden)
        prediction = self.fc_out(output[0])
        return prediction, hidden
    


class AttnDecoder(nn.Module):
    def __init__(self, rnn, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.attention = Attention(enc_hid_dim, dec_hid_dim)
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = rnn
        self.fc_out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, encoder_outputs):
        #input = [1, batch size]
        #hidden = (h, c) or h, [num layers, batch size, dec hid dim]
        #encoder_outputs = [src len, batch size, enc hid dim * 2]
        
        embedded = self.dropout(self.embedding(input)) # [1, batch size, emb dim]
        
        # compute attention weights using the hidden state of the first layer of decoder
        h = hidden[0][0] if isinstance(hidden, tuple) else hidden[0] # [batch size, dec hid dim]
        a = self.attention(h, encoder_outputs) # [batch size, src len]
        a = a.unsqueeze(1) # [batch size, 1, src len]
        encoder_outputs = encoder_outputs.permute(1, 0, 2) # [batch size, src len, enc hid dim * 2]
        
        weighted = torch.bmm(a, encoder_outputs) # [batch size, 1, enc hid dim * 2]
        weighted = weighted.permute(1, 0, 2) # [1, batch size, enc hid dim * 2]
        
        rnn_input = torch.cat((embedded, weighted), dim = 2) # [1, batch size, (enc hid dim * 2) + emb dim]

        output, hidden = self.rnn(rnn_input, hidden)
        #output = [1, batch size, dec hid dim * n directions]
        #hidden = [n layers * n directions, batch size, dec hid dim]
        
        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)
        
        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim = 1)) # [batch size, output dim]
        
        return prediction, hidden


class RNNModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        emb_dim = config.emb_dim
        hid_dim = config.hid_dim
        enc_layers = config.enc_layers
        dec_layers = config.dec_layers
        dropout = config.dropout
        use_attention = 'attn' in config.model

        # encoder
        if 'GRU' in self.config.model:
            self.encoder = nn.GRU(emb_dim, hid_dim, enc_layers, dropout=dropout, bidirectional=True)
            enc_out_dim = hid_dim * 2
        elif 'LSTM' in self.config.model:
            self.encoder = nn.LSTM(emb_dim, hid_dim, enc_layers, dropout=dropout, bidirectional=True)
            enc_out_dim = hid_dim * 2
        elif config.model == 'ON':
            self.encoder = ONLSTMStack([emb_dim] + [hid_dim] * enc_layers, chunk_size=8, dropconnect=dropout, dropout=dropout)
            enc_out_dim = hid_dim
        elif config.model == 'OM':
            self.encoder = OrderedMemory(emb_dim, hid_dim, nslot=21, bidirection=False)
            enc_out_dim = hid_dim
        else:
            pass

        self.map_h = nn.Linear(enc_out_dim, hid_dim)
        if config.model in ['LSTM', 'LSTM_attn', 'ON']:
            self.map_c = nn.Linear(enc_out_dim, hid_dim)

        # decoder
        if config.result_encoding == 'sin':
            self.decoder = SinDecoder(inp_dim=hid_dim, res_dim=emb_dim, feedforward_dims=[], dropout=dropout)
        else:
            dec_inp_dim = enc_out_dim + emb_dim if use_attention else emb_dim
            if config.model in ['LSTM', 'LSTM_attn', 'ON']:
                rnn = nn.LSTM(dec_inp_dim, hid_dim, dec_layers, dropout=dropout, bidirectional=False)
            elif config.model in ['GRU', 'GRU_attn', 'OM']:
                rnn = nn.GRU(dec_inp_dim, hid_dim, dec_layers, dropout=dropout, bidirectional=False)
            
            decoder_cls = AttnDecoder if use_attention else Decoder
            self.decoder = decoder_cls(rnn, config.out_vocab_size, emb_dim, hid_dim, hid_dim, dropout=dropout)


    def encode(self, src, src_len):
        encoder_outputs, h, c = None, None, None

        if 'GRU' in self.config.model:
            encoder_outputs, h = self.encoder(src)
            h = h.view(-1, 2, *h.shape[1:])[-1].transpose(0, 1)
            h = h.contiguous().view(h.shape[0], -1)
        elif 'LSTM' in self.config.model:
            encoder_outputs, (h, c) = self.encoder(src)
            # h: [num_layers*n_directions, batch_size, hid_dim]
            # c: [num_layers*n_directions, batch_size, hid_dim]
            h = h.view(-1, 2, *h.shape[1:])[-1].transpose(0, 1)
            h = h.contiguous().view(h.shape[0], -1)

            c = c.view(-1, 2, *c.shape[1:])[-1].transpose(0, 1)
            c = c.contiguous().view(c.shape[0], -1)

        elif self.config.model == 'ON':
            h, c = self.encoder(src)[1][-1]
            c = c.contiguous().view(c.shape[0], -1)

        elif self.config.model == 'OM':
            mask = torch.from_numpy(create_padding_mask(src_len)).to(src.device).transpose(0,1).contiguous()
            h = self.encoder(src, mask, output_last=True)

        h = torch.tanh(self.map_h(h))
        if c is not None:
            c = torch.tanh(self.map_c(c))
            hidden = (h, c)
        else:
            hidden = h

        return encoder_outputs, hidden

    def decode(self, encoder_outputs, hidden, tgt, teacher_forcing_ratio):
        if self.config.result_encoding == 'sin':
            if isinstance(hidden, tuple):
                hidden = hidden[0]
            return self.decoder(hidden)

        max_len, batch_size = tgt.shape
        decoder_input = torch.tensor([[self.config.decoder_sos] * batch_size], device=tgt.device)
        output_list = []
    
        if isinstance(hidden, tuple):
            h, c = hidden
            h = torch.stack([h] * self.config.dec_layers)
            c = torch.stack([c] * self.config.dec_layers)
            hidden = (h, c)
        else:
            h = hidden
            h = torch.stack([h] * self.config.dec_layers)
            hidden = h

        teacher_force = True if self.training and random.random() < teacher_forcing_ratio else False
        for i in range(max_len):
            output, hidden = self.decoder(decoder_input, hidden, encoder_outputs)
            if teacher_force and i < max_len - 1:
                decoder_input = tgt[i+1].view(1, -1)
            else:
                topv, topi = output.topk(1)
                decoder_input = topi.view(1, -1).detach()
            output_list.append(output)
        
        output = torch.stack(output_list)
        return output.transpose(0, 1)

    def forward(self, src, src_len, tgt, teacher_forcing_ratio=0.5, output_attentions=False):
        src = src.transpose(0, 1)
        tgt = tgt.transpose(0, 1)
        encoder_outputs, hidden = self.encode(src, src_len)
        output = self.decode(encoder_outputs, hidden, tgt, teacher_forcing_ratio)

        return output

