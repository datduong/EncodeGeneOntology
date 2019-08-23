

from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string, re, sys, os, pickle
from tqdm import tqdm
import numpy as np
from collections import namedtuple
from tempfile import TemporaryDirectory

import logging
import json

from scipy.special import softmax

import pandas as pd

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.init import xavier_uniform_

from torch.utils.data import DataLoader, Dataset, RandomSampler

from torch.nn import CrossEntropyLoss, MSELoss
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score


def simple_accuracy(preds, labels):
  return (preds == labels).mean()

def acc_and_f1(preds, labels, metric_option):
  if metric_option == 'entailment':
    preds = np.argmax(preds, axis=1)

  if metric_option == 'cosine':
    preds[preds<=0] = -1 # rounding at 0
    preds[preds>0] = 1

  acc = simple_accuracy(preds, labels)
  f1 = f1_score(y_true=labels, y_pred=preds)
  return {
      "acc": acc,
      "f1": f1,
      "acc_and_f1": (acc + f1) / 2,
  }


class cosine_distance_loss (nn.Module):
  def __init__(self, word_vec_dim, out_dim, args):

    super(cosine_distance_loss, self).__init__()

    self.args = args
    if self.args.reduce_cls_vec:
      self.reduce_vec_dim = nn.Linear(word_vec_dim, out_dim)
      xavier_uniform_(self.reduce_vec_dim.weight)

    # margin=-1 means, that when y=-1, then we want max(0, x- -1) = max(0, x+1) to be small.
    # for this function to be small, we have to get x --> -1.
    # self.loss = nn.CosineEmbeddingLoss(margin = -1) ## https://pytorch.org/docs/stable/nn.html#torch.nn.CosineEmbeddingLoss

    self.loss = nn.MSELoss()


  def forward(self,emb1,emb2,true_label): ## not work with fp16, so we have to write own own cosine distance ??

    if self.args.reduce_cls_vec:
      emb1 = self.reduce_vec_dim(emb1)
      emb2 = self.reduce_vec_dim(emb2)

    score = F.cosine_similarity(emb1,emb2,dim=1,eps=.0001) ## not work for fp16 ?? @score is 1 x batch_size
    loss = self.loss(score, true_label)

    # loss = self.loss(emb1, emb2, true_label)
    # with torch.no_grad():
    #   score = F.cosine_similarity(emb1,emb2,dim=1,eps=.0001) ## not work for fp16 ?? # @score is 1 x batch_size

    return loss, score

class encoder_model(nn.Module):

  def __init__(self, args, metric_model, **kwargs):
    super(encoder_model, self).__init__()

    self.metric_option = kwargs['metric_option'] ## should be 'entailment' or 'cosine'
    self.metric_module = metric_model
    self.args = args

    self.mode = 'text'

    if ".pickle" in self.args.vector_file:
      self.mode = 'pickle' ## onto2vec is saved as pickle, we can convert to text, but let's just use the pickle directly

    if self.mode == 'text':
      self.go = pd.read_csv(self.args.vector_file, sep='\t')
      non_formatted_go = self.go.set_index('name').T.to_dict('list')
      self.go_dict = {}
      for go_vecs in non_formatted_go: ## we are using dictionary to look-up, ordering does not matter
        split = non_formatted_go[go_vecs][0].split()
        float_vec = [float(b) for b in split]
        self.go_dict[go_vecs] = float_vec

    else: ## Onto2vec came in dictionary format.
      self.go_dict = {}
      go_dict = pickle.load(open(self.args.vector_file,"rb"))
      for key,val in go_dict.items(): 
        ## put into format GO:xyz
        if 'GO:' not in key: 
          key = "GO:"+key
        #
        self.go_dict [key] = val

  def forward(self, go_terms):
    ## some GO terms are too new or too old, must be removed ?? seems very stupid ???
    ## we will set them to 0. in the 1st attempt, it looks like this is ver rare
    go_vectors = [ ] 
    for a in go_terms : 
      if a in self.go_dict : 
        go_vectors.append ( self.go_dict[a] )
      else:
        go_vectors.append ( [0]*self.args.def_emb_dim ) 
    return torch.FloatTensor (go_vectors) ## 2D array, must be converted to tensor for CUDA

  def convertToString(self,label_names):
    act_label_names = []
    for go_num in label_names:
      st = str(go_num.item())
      st_len = len(st)
      for i in range(st_len, 7):
        st = '0' + st
      st = 'GO:' + st
      act_label_names.append(st)
    return act_label_names

  def do_eval(self, train_dataloader):
    torch.cuda.empty_cache()
    self.eval()

    tr_loss = 0
    preds = []
    all_label_ids = []

    ## for each batch
    for step, batch in enumerate(tqdm(train_dataloader, desc="eval")):

      with torch.no_grad():
        batch = tuple(t for t in batch)

        label_one_names, _ , _ , _ , label_two_names, _ , _ , _ , label_ids = batch

        actual_one_names = self.convertToString(label_one_names)
        actual_two_names = self.convertToString(label_two_names)
        label_vec_left = self.forward(actual_one_names)
        label_vec_right = self.forward(actual_two_names)

        loss, score = self.metric_module.forward(label_vec_left.cuda(), label_vec_right.cuda(), true_label=label_ids.cuda())

      tr_loss = tr_loss + loss

      if len(preds) == 0:
        preds.append(score.detach().cpu().numpy())
        all_label_ids.append(label_ids.detach().cpu().numpy())
      else:
        preds[0] = np.append(preds[0], score.detach().cpu().numpy(), axis=0)
        all_label_ids[0] = np.append(all_label_ids[0], label_ids.detach().cpu().numpy(), axis=0) # row array

    # end eval
    all_label_ids = all_label_ids[0]
    preds = preds[0]

    if self.metric_option == 'entailment':
      preds = softmax(preds, axis=1) ## softmax, return both prob of 0 and 1 for each label

    print (preds)
    print (all_label_ids)

    result = 0
    if self.args.test_file is None: ## save some time
      result = acc_and_f1(preds, all_label_ids, self.metric_option) ## interally, we will take care of the case of @entailment vs @cosine
      for key in sorted(result.keys()):
        print("%s=%s" % (key, str(result[key])))

    return result, preds, tr_loss

  def write_label_vector (self,label_desc_loader,fout_name,label_name):

    self.eval()

    if fout_name is not None:
      fout = open(fout_name,'w')

    label_emb = None

    counter = 0 ## count the label to be written
    for step, batch in enumerate(tqdm(label_desc_loader, desc="write label desc")):

      batch = tuple(t for t in batch)

      label_names1, label_desc1, label_len1, _ = batch

      with torch.no_grad():
        label_desc1.data = label_desc1.data[ : , 0:int(max(label_len1)) ] # trim down input to max len of the batch
        actual_one_names = self.convertToString(label_names1)
        label_emb1 = self.forward(actual_one_names)
        if self.args.reduce_cls_vec:
          label_emb1 = self.metric_module.reduce_vec_dim(label_emb1)

      label_emb1 = label_emb1.detach().cpu().numpy()

      if fout_name is not None:
        for row in range ( label_emb1.shape[0] ) :
          fout.write( label_name[counter] + "\t" + "\t".join(str(m) for m in label_emb1[row]) + "\n" )
          counter = counter + 1

      if label_emb is None:
        label_emb = label_emb1
      else:
        label_emb = np.concatenate((label_emb, label_emb1), axis=0) ## so that we have num_go x dim

    if fout_name is not None:
      fout.close()

    return label_emb
