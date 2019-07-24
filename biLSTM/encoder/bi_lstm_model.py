
from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string, re, sys
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
# from torch.nn.init import xavier_uniform_


class bi_lstm_sent_encoder (nn.Module):
	def __init__(self, lstm_input_dim, lstm_out_dim, use_cuda=True, rnn_type='LSTM', dropout=0.1, num_layers=1): # @rnn_type can be GRU
		super().__init__()
		self.lstm_out_dim = lstm_out_dim

		## doing this will avoid having to inti the @hidden_state_time_zero
		## here, @lstm_out_dim will be *2 because of bidirection
		self.lstm = getattr(nn, rnn_type)(lstm_input_dim, lstm_out_dim//2, num_layers=num_layers, ## @1 for 1 layer
																		batch_first=True, dropout=dropout,
																		bidirectional=True)
		self.use_cuda = use_cuda

	def forward(self, embeds, seq_lengths): ## @seq_lengths is used to avoid useless lstm passing to end of padding.

		# embeds = self.drop ( embeds )

		# @embeds must be sorted by len
		# for padding, we need to sort by len
		seq_lengths, idx_sort = np.sort(seq_lengths)[::-1].copy(), np.argsort(-seq_lengths) ## @.copy() is needed to avoid "uncontiguous" block of numbers
		idx_unsort = np.argsort(idx_sort)

		if self.use_cuda:
			idx_sort = Variable(idx_sort).cuda()  # torch.from_numpy(idx_sort) #.cuda()
		else:
			idx_sort = Variable(idx_sort)

		embeds = embeds.index_select(0, idx_sort ) ## order the embedding by len

		# Handling padding in Recurrent Networks
		packed_input = pack_padded_sequence(embeds, seq_lengths, batch_first=True)
		##  REMEMBER TO CLEAR THE HIDDEN_STATE OTHERWISE, WILL SEE OPTIM ERROR
		lstm_out, _ = self.lstm(packed_input)
		# unpack your output if required
		lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True, padding_value=-np.inf) ## see the zero as padding

		# Un-sort by length, so we get back the original ordering not sorted by len.
		# idx_unsort = torch.from_numpy(idx_unsort) #.cuda()
		if self.use_cuda :
			lstm_out = lstm_out.index_select(0, Variable(idx_unsort).cuda() )
		else:
			lstm_out = lstm_out.index_select(0, Variable(idx_unsort) )

		lstm_out , _ = torch.max (lstm_out, dim=1 ) # num_batch x lstm_dim
		return lstm_out


