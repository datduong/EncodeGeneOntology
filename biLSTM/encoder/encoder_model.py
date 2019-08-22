

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

import pandas as pd

import biLSTM.encoder.bi_lstm_model as bi_lstm_model

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


class ReadGOVecFromFile(nn.Module):

  def __init__(self, go_text, dim=300):
    super(ReadGOVecFromFile, self).__init__()

    self.dim = dim ## dim of pretrained, should be 300 to match the other encoders
    self.mode = 'text'
    if ".pickle" in go_text:
      self.mode = 'pickle' ## onto2vec is saved as pickle, we can convert to text, but let's just use the pickle directly

    if self.mode == 'text':
      self.go = pd.read_csv(go_text, sep='\t')
      non_formatted_go = self.go.set_index('name').T.to_dict('list')
      self.go_dict = {}
      for go_vecs in non_formatted_go:
        split = non_formatted_go[go_vecs][0].split()
        float_vec = [float(b) for b in split]
        self.go_dict[go_vecs] = float_vec

    else:
      self.go_dict = pickle.load(open(go_text,"rb"))

  def forward(self, go_terms):
    ## @go_terms makes sure this variable matches the GO names array in BiLSTM, GCN or BERT. Ordering matter here.
    ## because of strict enforce GO:xyz syntax to input @go_terms, we may need to add in "GO:"
    # go_vectors = [self.go_dict[a] for a in go_terms]
    go_vectors = np.zeros((len(go_terms),self.dim))

    for index, go in enumerate(go_terms):
      if go not in self.go_dict:
        go_vectors [index] = self.go_dict[re.sub("GO:","",go)] ## because of strict enforce GO:xyz syntax, the pickle may not have the "GO:"
      else:
        go_vectors [index] = self.go_dict[go]

    return go_vectors ## probably doesn't need to be in pytorch tensor


class encoder_model (nn.Module) :

  def __init__(self,args,metric_module,biLstm,**kwargs):

    # metric_module is either @entailment_model or cosine distance.
    # we observe that entailment_model doesn't directly ensure the same labels to have the same vectors. entailment_model pass concatenate v1,v2,v1*v2,abs(v1-v2) into an MLP

    super(encoder_model, self).__init__()

    self.metric_option = kwargs['metric_option'] ## should be 'entailment' or 'cosine'
    self.metric_module = metric_module

    self.args = args

    self.word_embedding = nn.Embedding(kwargs['num_of_word'],kwargs['word_vec_dim']) ## word embedding
    if 'pretrained_weight' in kwargs:
      self.word_embedding.weight.data.copy_(torch.from_numpy(kwargs['pretrained_weight']))

    self.biLstm = biLstm
    self.dropout = nn.Dropout (kwargs['dropout'])

    self.optimizer = None


  def bilstm_layer (self,label_idx,label_len):
    label_emb = self.word_embedding(label_idx) # get word vector for each label
    label_emb = self.dropout(label_emb)
    return self.biLstm.forward(label_emb,label_len) # one vector for each label, so batch x dim

  def make_optimizer (self):
    if self.args.fix_word_emb:
      return torch.optim.Adam ( [p for n,p in self.named_parameters () if "word_embedding" not in n] , lr=self.args.lr )
    else:
      return torch.optim.Adam ( self.parameters(), lr=self.args.lr )

  def do_train(self,train_dataloader,dev_dataloader=None):

    torch.cuda.empty_cache()

    optimizer = self.make_optimizer()

    eval_acc = 0
    lowest_dev_loss = np.inf

    for epoch in range( int(self.args.epoch)) :

      self.train()
      tr_loss = 0

      ## for each batch
      for step, batch in enumerate(tqdm(train_dataloader, desc="ent. epoch {}".format(epoch))):

        batch = tuple(t for t in batch)

        label_desc1, label_len1, label_mask1, label_desc2, label_len2, label_mask2, label_ids = batch

        label_vec_left = self.bilstm_layer (label_desc1.cuda(),label_len1)
        label_vec_right = self.bilstm_layer (label_desc2.cuda(),label_len2)

        ## need to backprop somehow
        ## predict the class bio/molec/cellcompo ?
        ## predict if 2 labels are similar ? ... sort of doing the same thing as gcn already does
        loss, _ = self.metric_module.forward(label_vec_left, label_vec_right, true_label=label_ids.cuda())

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        tr_loss = tr_loss + loss

      ## end epoch
      print ("\ntrain epoch {} loss {}".format(epoch,tr_loss))

      # eval at each epoch
      # print ('\neval on train data epoch {}'.format(epoch))
      # result, _ , _ = self.do_eval(train_dataloader,labeldesc_loader,edge_index)

      print ('\neval on dev data epoch {}'.format(epoch))
      result, preds, dev_loss = self.do_eval(dev_dataloader)

      if dev_loss < lowest_dev_loss :
        lowest_dev_loss = dev_loss
        print ("save best, lowest dev loss {}".format(lowest_dev_loss))
        torch.save(self.state_dict(), os.path.join(self.args.result_folder,"best_state_dict.pytorch"))
        last_best_epoch = epoch

      if epoch - last_best_epoch > 20:
        print ('\n\n\n**** break early \n\n\n')
        print ('')
        return tr_loss

    return tr_loss ## last train loss

  def do_eval(self,train_dataloader):

    torch.cuda.empty_cache()
    self.eval()

    tr_loss = 0
    preds = []
    all_label_ids = []

    ## for each batch
    for step, batch in enumerate(tqdm(train_dataloader, desc="eval")):

      with torch.no_grad():
        batch = tuple(t for t in batch)

        label_desc1, label_len1, label_mask1, label_desc2, label_len2, label_mask2, label_ids = batch

        label_vec_left = self.bilstm_layer (label_desc1.cuda(),label_len1)
        label_vec_right = self.bilstm_layer (label_desc2.cuda(),label_len2)

        loss, score = self.metric_module.forward(label_vec_left, label_vec_right, true_label=label_ids.cuda())


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

      label_desc1, label_len1, _ = batch

      with torch.no_grad():
        label_desc1.data = label_desc1.data[ : , 0:int(max(label_len1)) ] # trim down input to max len of the batch

        label_emb1 = self.bilstm_layer(label_desc1.cuda(),label_len1)
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



