

from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string, re, sys, os
from tqdm import tqdm
import numpy as np
from collections import namedtuple
from tempfile import TemporaryDirectory

from copy import deepcopy

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

from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule


sys.path.append("/local/datdb/GOmultitask")
import GCN.encoder.bi_lstm_model as bi_lstm_model
import BERT.encoder.encoder_model as BERT_encoder_model

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


class encoder_model (nn.Module) :

  def __init__(self,args,metric_module,**kwargs):

    # metric_module is either @entailment_model or cosine distance.
    # we observe that entailment_model doesn't directly ensure the same labels to have the same vectors. entailment_model pass concatenate v1,v2,v1*v2,abs(v1-v2) into an MLP

    super(encoder_model, self).__init__()

    # able to call BERT, but we will not be able to encode BERT for many nodes at once.
    # self.tokenizer = tokenizer
    # self.bert_lm_sentence = bert_lm_sentence ## bert LM already tuned model

    self.metric_option = kwargs['metric_option'] ## should be 'entailment' or 'cosine'
    self.metric_module = metric_module

    self.args = args

    self.gcn1 = GCNConv(args.def_emb_dim, args.gcnn_dim)
    self.gcn2 = GCNConv(args.gcnn_dim, args.gcnn_dim)

    self.label_embedding = nn.Embedding(args.num_label,args.def_emb_dim) ## each label is a vector

    self.classify_loss = nn.CrossEntropyLoss()

    self.nonlinear_gcnn = kwargs['nonlinear_gcnn']

    self.dropout = nn.Dropout (kwargs['dropout'])

    self.optimizer = None

  # def do_gcn(self,input_idx,edge_index): ## @input_idx is simple label indexing [0 10 5] = take in label #0 #10 #5
  #   label_emb = self.label_embedding(input_idx) ## batch x sent_len x dim
  #   return self.gcn (label_emb,edge_index)

  def gcn_2layer (self,labeldesc_loader,edge_index):

    node_emb = self.nonlinear_gcnn ( self.gcn1.forward ( self.dropout ( self.label_embedding.weight ), edge_index) ) ## take in entire label space at once
    return self.gcn2.forward (node_emb, edge_index) ## not relu or tanh in last layer

  def make_optimizer (self):
    if self.args.fix_word_emb:
      print ([n for n,p in self.named_parameters () if "label_embedding" not in n])
      return torch.optim.Adam ( [p for n,p in self.named_parameters () if "label_embedding" not in n] , lr=self.args.lr )
    else:
      return torch.optim.Adam ( self.parameters(), lr=self.args.lr )

  def do_train(self,train_dataloader,labeldesc_loader,edge_index,dev_dataloader=None):

    torch.cuda.empty_cache()

    optimizer = self.make_optimizer()

    eval_acc = 0
    lowest_dev_loss = np.inf

    for epoch in range( int(self.args.epoch)) :

      self.train()
      tr_loss = 0

      ## for each batch
      for step, batch in enumerate(tqdm(train_dataloader, desc="ent. epoch {}".format(epoch))):

        label_emb = self.gcn_2layer(labeldesc_loader,edge_index)

        batch = tuple(t for t in batch)

        label_id_number_left, label_id_number_right, label_ids = batch

        label_id_number_left = label_id_number_left.squeeze(1).data.numpy() ## using as indexing, so have to be array int, not tensor
        label_id_number_right = label_id_number_right.squeeze(1).data.numpy()

        ## need to backprop somehow
        ## predict the class bio/molec/cellcompo ?
        ## predict if 2 labels are similar ? ... sort of doing the same thing as gcn already does
        loss, _ = self.metric_module.forward(label_emb[label_id_number_left], label_emb[label_id_number_right], true_label=label_ids.cuda())

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        tr_loss = tr_loss + loss

      ## end epoch

      # eval at each epoch
      # print ('\neval on train data epoch {}'.format(epoch))
      # result, _ , _ = self.do_eval(train_dataloader,labeldesc_loader,edge_index)

      print ('\neval on dev data epoch {}'.format(epoch))
      result, preds, dev_loss = self.do_eval(dev_dataloader,labeldesc_loader,edge_index)

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

  def do_eval(self,train_dataloader,labeldesc_loader,edge_index):

    torch.cuda.empty_cache()
    self.eval()

    tr_loss = 0
    preds = []
    all_label_ids = []

    with torch.no_grad(): ## don't need to update labels anymore
      label_emb = self.gcn_2layer(labeldesc_loader,edge_index)
      print ('sample gcn label_emb')
      print (label_emb)

    ## for each batch
    for step, batch in enumerate(tqdm(train_dataloader, desc="eval")):

      batch = tuple(t for t in batch)

      label_id_number_left, label_id_number_right, label_ids = batch

      label_id_number_left = label_id_number_left.squeeze(1).data.numpy() ## using as indexing, so have to be array int, not tensor
      label_id_number_right = label_id_number_right.squeeze(1).data.numpy()

      ## need to backprop somehow
      ## predict the class bio/molec/cellcompo ?
      ## predict if 2 labels are similar ? ... sort of doing the same thing as gcn already does
      with torch.no_grad():
        loss, score = self.metric_module.forward(label_emb[label_id_number_left], label_emb[label_id_number_right], true_label=label_ids.cuda())

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

  def write_label_vector (self,labeldesc_loader,edge_index,fout_name,label_name):
    self.eval()

    if fout_name is not None:
      fout = open(fout_name,'w')
      if self.args.reduce_cls_vec:
        fout.write(str(len(label_name)) + " " + str(300) + "\n")
      else:
        fout.write(str(len(label_name)) + " " + str(768) + "\n")

    label_emb = self.gcn_2layer(labeldesc_loader,edge_index) ##!! only need to call graph-forward-pass 

    if fout_name is not None:
      for row in range ( label_emb.shape[0] ) :
        fout.write( label_name[counter] + " " + " ".join(str(m) for m in label_emb[row]) + "\n" ) ## space, because gensim format
        counter = counter + 1
      #
      fout.close()

    return label_emb


class encoder_model_conv1d (encoder_model):

  def __init__(self,args,metric_module,**kwargs):

    super(encoder_model_conv1d, self).__init__(args,metric_module,**kwargs)

    self.dropout = nn.Dropout(p=kwargs['dropout'])
    self.conv1d = nn.Conv1d(args.word_emb_dim,args.word_emb_dim,stride=1,kernel_size=7,padding=3) ## retain same size output

    if 'pretrained_weight' in kwargs:
      ## !! MAKE SURE PADDING HAS 0 VECTOR VALUE
      # self.label_embedding = nn.EmbeddingBag(kwargs['num_of_word'],kwargs['word_vec_dim'],mode="sum")
      self.label_embedding = nn.Embedding(kwargs['num_of_word'],kwargs['word_vec_dim'],padding_idx=0)
      self.label_embedding.weight.data.copy_(torch.from_numpy(kwargs['pretrained_weight']))
    else:
      # self.label_embedding = nn.EmbeddingBag(args.num_vocab,args.word_emb_dim,mode="sum") ## word embedding
      self.label_embedding = nn.Embedding(kwargs['num_of_word'],kwargs['word_vec_dim'],padding_idx=0)

  def cnn_pool (self,labeldesc_loader):

    label_emb = torch.zeros(self.args.num_label,self.args.def_emb_dim).cuda()

    start = 0
    for step, batch in enumerate(labeldesc_loader):

      batch = tuple(t for t in batch)

      label_idx, label_len, mask = batch
      label_idx.data = label_idx.data[ : , 0:int(max(label_len)) ] ## trim down to max len in this batch
      mask.data = mask.data[ : , 0:int(max(label_len)) ] ## trim down to max len in this batch

      label_idx = label_idx.cuda()

      # if we use @EmbeddingBag, then we get word vector THAT ARE SUM TOGETHER ? @label_idx will be batch_size x word_vec_dim
      # label_idx = label_idx / label_len.squeeze(1) # take average

      label_idx = self.label_embedding(label_idx) ## get simple word embedding

      label_idx = self.conv1d (label_idx.transpose(1,2)) ## must transpose otherwise size error
      label_idx = F.relu(label_idx.transpose(1,2)) ## some non linear, transpose back to batch x sent_len x dim

      ## need masking, because each GO term have different length
      label_idx.data [ mask.data==0 ] = -np.inf ## so when we take max, we are ok
      label_idx, _ = torch.max (label_idx, 1) ## input is batch x sent_len x word_dim

      end = start + label_len.shape[0]
      label_emb[start:end] = label_idx # replace
      start = end

    return label_emb

  def gcn_2layer (self,labeldesc_loader,edge_index):

    label_emb = self.cnn_pool (labeldesc_loader) ## use label desc
    node_emb = self.nonlinear_gcnn ( self.gcn1.forward (label_emb, edge_index) ) ## take in entire label space at once
    return self.gcn2.forward (node_emb, edge_index) ## not relu or tanh in last layer


class encoder_model_avepool (encoder_model):

  def __init__(self,args,metric_module,**kwargs):

    super(encoder_model_avepool, self).__init__(args,metric_module,**kwargs)

    if 'pretrained_weight' in kwargs:
      ## !! MAKE SURE PADDING HAS 0 VECTOR VALUE
      self.label_embedding = nn.EmbeddingBag(kwargs['num_of_word'],kwargs['word_vec_dim'],mode="sum")
      self.label_embedding.weight.data.copy_(torch.from_numpy(kwargs['pretrained_weight']))
    else:
      self.label_embedding = nn.EmbeddingBag(args.num_vocab,args.word_emb_dim,mode="sum") ## word embedding

  def ave_pool (self,labeldesc_loader):

    label_emb = torch.zeros(self.args.num_label,self.args.def_emb_dim).cuda()

    start = 0
    for step, batch in enumerate(labeldesc_loader):

      batch = tuple(t for t in batch)

      label_idx, label_len, _ = batch
      label_idx.data = label_idx.data[ : , 0:int(max(label_len)) ] ## trim down to max len in this batch

      label_idx = label_idx.cuda()
      label_len = label_len.cuda()

      # if we use @EmbeddingBag, then we get word vector THAT ARE SUM TOGETHER ? @label_idx will be batch_size x word_vec_dim
      label_idx = self.label_embedding(label_idx) ## get sum of word embedding for each sent
      label_idx = label_idx / label_len.unsqueeze(1) # take average

      end = start + label_len.shape[0]
      label_emb[start:end] = label_idx # replace
      start = end

    return label_emb

  def gcn_2layer (self,labeldesc_loader,edge_index):

    label_emb = self.ave_pool (labeldesc_loader) ## use label desc
    node_emb = self.nonlinear_gcnn ( self.gcn1.forward (label_emb, edge_index) ) ## take in entire label space at once
    return self.gcn2.forward (node_emb, edge_index) ## not relu or tanh in last layer


class encoder_model_biLstm (encoder_model):

  def __init__(self,args,metric_module,biLstm,**kwargs):

    super(encoder_model_biLstm, self).__init__(args,metric_module,**kwargs)
    self.biLstm = biLstm
    self.dropout = nn.Dropout(p=kwargs['dropout'])

    self.reduce_vec_dim = nn.Linear(args.bilstm_dim, args.def_emb_dim) ## take something like 1024 reduce to 300
    xavier_uniform_(self.reduce_vec_dim.weight)

    if 'pretrained_weight' in kwargs:
      self.label_embedding = nn.Embedding(kwargs['num_of_word'],kwargs['word_vec_dim'],padding_idx=0)
      self.label_embedding.weight.data.copy_(torch.from_numpy(kwargs['pretrained_weight']))
    else:
      self.label_embedding = nn.Embedding(args.num_vocab,args.word_emb_dim,padding_idx=0) ## word embedding

  def lstm_layer (self,labeldesc_loader):
    ## take each batch of label desc, do biLstm, gather batch, send to gcnn.

    label_emb = torch.zeros(self.args.num_label,self.args.def_emb_dim).cuda()

    start = 0
    for step, batch in enumerate(labeldesc_loader):

      batch = tuple(t for t in batch)

      label_idx, label_len, _ = batch
      label_idx.data = label_idx.data[ : , 0:int(max(label_len)) ] ## trim down to max len in this batch

      label_idx = label_idx.cuda()
      label_idx = self.label_embedding(label_idx) # get word vector

      label_idx = self.dropout(label_idx)
      label_idx = self.biLstm.forward(label_idx,label_len) # one vector for each node

      end = start + label_len.shape[0]
      label_emb[start:end] = self.reduce_vec_dim (label_idx) # replace
      start = end

    return label_emb

  def gcn_2layer (self,labeldesc_loader,edge_index):

    label_emb = self.lstm_layer (labeldesc_loader) ## use label desc
    node_emb = self.nonlinear_gcnn ( self.gcn1.forward (label_emb, edge_index) ) ## take in entire label space at once
    return self.gcn2.forward (node_emb, edge_index) ## not relu or tanh in last layer


class encoder_model_extended_embedding (encoder_model):

  def __init__(self,args,metric_module, **kwargs):

    # add some vector like Onto2vec BiLSTM BERT into GCN

    super(encoder_model_extended_embedding, self).__init__(args,metric_module,**kwargs)

    if 'pretrained_weight' in kwargs:
      self.label_embedding = nn.Embedding(kwargs['num_of_word'],kwargs['word_vec_dim'])
      self.label_embedding.weight.data.copy_(torch.from_numpy(kwargs['pretrained_weight']))
      # freeze emb by doing @fix_word_emb
      # No need... self.label_embedding.requires_grad=False ## turn of gradient here ?
    else:
      print('\n\nERROR: Must provide pretrained_weight for this model\n\n')
      exit()

    self.gcn1 = GCNConv(args.def_emb_dim + args.gcn_native_emb_dim , args.gcnn_dim) # args.def_emb_dim + args.gcn_native_emb_dim
    self.gcn2 = GCNConv(args.gcnn_dim, args.gcnn_dim)

    # self.LinearCombine = nn.Linear(args.def_emb_dim+args.gcnn_dim, args.gcnn_dim)

    self.dropout = nn.Dropout(p=kwargs['dropout'])

    ## let gcn capture information not found in BERT or BiLSTM or whatever.
    self.gcn_native_emb = nn.Embedding(args.num_label,args.gcn_native_emb_dim)
    self.gcn_native_emb.weight.data.normal_(mean=0.0, std=0.2)

  def gcn_2layer (self,labeldesc_loader,edge_index):

    combined_embed = torch.cat((self.label_embedding.weight, self.gcn_native_emb.weight), 1)
    node_emb = self.nonlinear_gcnn ( self.gcn1.forward ( self.dropout ( combined_embed ), edge_index) ) ## take in entire label space at once
    node_emb = self.gcn2.forward (node_emb, edge_index) ## not relu or tanh in last layer

    ## try concat @label_embedding after we call gcn
    # node_emb = self.nonlinear_gcnn ( self.gcn1.forward ( self.dropout ( self.gcn_native_emb.weight ), edge_index) )
    # node_emb = self.gcn2.forward (node_emb, edge_index) ## not relu or tanh in last layer
    # node_emb = torch.cat((self.label_embedding.weight, node_emb), 1)
    # node_emb = self.LinearCombine(node_emb)

    return node_emb


class encoder_with_bert (encoder_model): # add some vector like Onto2vec BiLSTM BERT into GCN

  def __init__(self,args,BertModel,metric_module, **kwargs):
    super(encoder_with_bert, self).__init__(args,metric_module,**kwargs)
    self.bert = BertModel ## something like bert(GOdef) = GOvec
    self.linear_reduce_bert = nn.Linear(768,512) ## bert is 768, if we don't want to retrain it, then we need to use 768-->512, then appen 512 to GCN

  def make_optimizer (self,num_train_optimization_steps):  ## use bert style optimizer

    param_optimizer = list( self.named_parameters() ) ## all the params
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
      {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)] , 'weight_decay': 0.01},
      {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)] , 'weight_decay': 0.0}
      ]

    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=self.args.learning_rate,
                         warmup=self.args.warmup_proportion,
                         t_total=num_train_optimization_steps)
    return optimizer


  def get_label_gcn (self,batch,edge_index):

    label_emb_gcn = self.gcn_2layer(None, edge_index) ## @labeldesc_loader was mean to be at None, but we don't care about label loader if we just use simple emb.
    label_id_number_left, label_id_number_right, _, _, _, _, _, _, _ = batch
    label_id_number_left = label_id_number_left.squeeze(1).data.numpy() ## using as indexing, so have to be array int, not tensor
    label_id_number_right = label_id_number_right.squeeze(1).data.numpy()
    return label_emb_gcn[label_id_number_left] , label_emb_gcn[label_id_number_right]

  def get_label_bert (self,batch):

    _, _, label_desc1, label_len1, label_mask1, label_desc2, label_len2, label_mask2, _ = batch

    label_desc1.data = label_desc1.data[ : , 0:int(max(label_len1)) ] # trim down input to max len of the batch
    label_mask1.data = label_mask1.data[ : , 0:int(max(label_len1)) ] # trim down input to max len of the batch
    label_emb1 = self.bert.encode_label_desc(label_desc1.cuda(),label_len1.cuda(),label_mask1.cuda())

    label_desc2.data = label_desc2.data[ : , 0:int(max(label_len2)) ]
    label_mask2.data = label_mask2.data[ : , 0:int(max(label_len2)) ]
    label_emb2 = self.bert.encode_label_desc(label_desc2.cuda(),label_len2.cuda(),label_mask2.cuda())

    return self.linear_reduce_bert( label_emb1 ) , self.linear_reduce_bert ( label_emb2 )

  def do_train(self,train_dataloader,labeldesc_loader,edge_index,num_train_optimization_steps,dev_dataloader=None):

    torch.cuda.empty_cache()

    optimizer = self.make_optimizer(num_train_optimization_steps)

    global_step = 0
    eval_acc = 0
    lowest_dev_loss = np.inf

    for epoch in range( int(self.args.epoch)) :

      torch.cuda.empty_cache()

      self.train()
      tr_loss = 0

      ## for each batch
      for step, batch in enumerate(tqdm(train_dataloader, desc="ent. epoch {}".format(epoch))):

        batch = tuple(t for t in batch)

        _, _, _, _, _, _, _, _, label_ids = batch ## seem so stupid

        label_gcn1, label_gcn2 = self.get_label_gcn (batch,edge_index)
        label_bert1, label_bert2 = self.get_label_bert (batch)

        loss, _ = self.metric_module.forward(
          torch.cat((label_gcn1,label_bert1),1),
          torch.cat((label_gcn2,label_bert2),1),
          true_label=label_ids.cuda())

        if self.args.gradient_accumulation_steps > 1:
          loss = loss / self.args.gradient_accumulation_steps

        loss.backward()
        tr_loss = tr_loss + loss

        if (step + 1) % self.args.gradient_accumulation_steps == 0:
          optimizer.step()
          optimizer.zero_grad()
          global_step += 1

      ## end epoch

      # eval at each epoch
      # print ('\neval on train data epoch {}'.format(epoch))
      # result, _ , _ = self.do_eval(train_dataloader,labeldesc_loader,edge_index)

      print ('\neval on dev data epoch {}'.format(epoch))
      result, preds, dev_loss = self.do_eval(dev_dataloader,labeldesc_loader,edge_index)

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

  def do_eval(self,train_dataloader,labeldesc_loader,edge_index):

    torch.cuda.empty_cache()
    self.eval()

    tr_loss = 0
    preds = []
    all_label_ids = []

    with torch.no_grad(): ## don't need to update labels anymore
      label_emb = self.gcn_2layer(labeldesc_loader,edge_index)
      print ('sample gcn label_emb')
      print (label_emb)

    ## for each batch
    for step, batch in enumerate(tqdm(train_dataloader, desc="eval")):

      batch = tuple(t for t in batch)

      with torch.no_grad():
        _, _, _, _, _, _, _, _, label_ids = batch ## seem so stupid

        label_gcn1, label_gcn2 = self.get_label_gcn (batch,edge_index)
        label_bert1, label_bert2 = self.get_label_bert (batch)

        loss, score = self.metric_module.forward(
          torch.cat((label_gcn1,label_bert1),1),
          torch.cat((label_gcn2,label_bert2),1),
          true_label=label_ids.cuda())

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

    torch.cuda.empty_cache()
    return result, preds, tr_loss











