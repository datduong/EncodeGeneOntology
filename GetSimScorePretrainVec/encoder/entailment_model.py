


from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string, re, sys

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.init import xavier_uniform_


class entailment_model (nn.Module) :
  def __init__(self,num_label_class,word_vec_dim,final_dim,weight=None,**kwargs): # @hidden_dim is dim of output after the CNN layer.
    super(entailment_model, self).__init__()

    self.num_label_class = num_label_class
    self.final_dim = final_dim
   
    # self.reduce_vec_dim = nn.Sequential(nn.Linear(word_vec_dim, final_dim),  # convert input vector to have new dim
    #                                     nn.Tanh())

    self.reduce_vec_dim = nn.Linear( word_vec_dim, final_dim )

    first_in_dim = final_dim*4

    self.distance_metric = nn.Sequential (
      nn.Linear(first_in_dim, first_in_dim//2 ) ,
      # nn.Dropout(kwargs['dropout']),
      nn.ReLU(),
      nn.Linear(first_in_dim//2, first_in_dim//4) ,
      # nn.Dropout(kwargs['dropout']),
      nn.ReLU(),
      nn.Linear(first_in_dim//4, self.num_label_class)
      ) ## entailment or not, so 2 scores

    self.nn_loss_function = nn.CrossEntropyLoss(weight=weight)  # torch.FloatTensor([0.75,1.5])

  def forward ( self, sent1_emb, sent2_emb, true_label=None ): # @sent1_emb are batch num_batch x num_word x max_length

    ## !! already used some LSTM outside @entailment_model. so that @sent1_emb is already vec emb of the def.

    sent1_emb = self.reduce_vec_dim (sent1_emb)
    sent2_emb = self.reduce_vec_dim (sent2_emb)

    angle = torch.mul ( sent1_emb, sent2_emb ) ## want num_batch x 1... this is this pointwise ??

    distance = torch.abs(sent1_emb - sent1_emb)

    combined_representation = torch.cat((sent1_emb, sent2_emb, angle, distance), dim=1) ## num_batch x 1 x rep_length (must be 2*(lstm_out + appended_neighbor) + other_factor )

    score = self.distance_metric.forward(combined_representation)

    if true_label is None:
      return 0, score ## loss is 0
    else:
      ## nn_loss_function see https://pytorch.org/docs/stable/nn.html#crossentropyloss
      return self.nn_loss_function (score, true_label), score



