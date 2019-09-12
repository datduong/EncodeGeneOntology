

from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string, re, sys, os, math
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

sys.path.append('/local/datdb/ProteinEmbMethodGithub/protein-sequence-embedding-iclr2019/')
from src.alphabets import Uniprot21
import src.scop as scop
from src.utils import pack_sequences, unpack_sequences
from src.utils import PairedDataset, AllPairsDataset, collate_paired_sequences
from src.utils import MultinomialResample
import src.models.embedding
import src.models.comparison

import fmax

from scipy.special import expit

from sklearn.metrics import f1_score

from torch_geometric.nn import GCNConv

import evaluation_metric

## simple model to predict GO for protein sequences

class ConcatCompare (nn.Module):
  def __init__(self,dim_in,dim_out,**kwargs):

    # @dim_in is size of @pointwise_interaction (see below)
    # @dim_out is 1, if we want to predict yes/no ... or @dim_out = something if we return a vector

    super().__init__()

    ## do we need 3 layers ? probably depend on complexity of the problem.

    self.CompareVec = nn.Sequential(nn.Linear(dim_in, dim_in//2),
                                    nn.Tanh(),
                                    nn.Dropout(kwargs['dropout']),
                                    nn.Linear(dim_in//2, dim_in//4),
                                    nn.Tanh(),
                                    nn.Dropout(kwargs['dropout']),
                                    nn.Linear(dim_in//4, dim_out))

    ## good initial values for optim
    for i in [0,3,6]:
      xavier_uniform_(self.CompareVec[i].weight)

  def forward(self,vec1,vec2):
    ## must make sure @vec1 @vec2 have proper dim.
    ## @vec1 @vec2 can be 2D matrix or 3D matrix

    ## takes 2 vectors x, y (same dimension)
    ## do x*y (pointwise) then abs(x-y), then concat to x,y
    pointwise_interaction = vec1 * vec2
    abs_diff = torch.abs(vec1-vec2)
    combine_vec = torch.cat ((vec1,vec2,pointwise_interaction,abs_diff), dim = 2) ## batch x num_go x dim

    return self.CompareVec(combine_vec).squeeze(2) ## make into 2D


class maxpoolNodeLayer (nn.Module):
  ## take maximum prob over [ node , children(node) ]
  def __init__(self,AdjacencyMatrix):
    super().__init__()
    self.AdjacencyMatrix = AdjacencyMatrix.cuda() ## type Long to use as indexing

  def forward(self,prob):
    ## @input is batch x dim
    ## call @input.unsqueeze(1) makes input into batch x 1 x dim, so we can do broadcast multiplication
    # self.AdjacencyMatrix * input.unsqueeze(1) gives batch x adjacency_row x len_input (here len_input = adjacency_row)
    val , _ = torch.max( self.AdjacencyMatrix * prob.unsqueeze(1), dim = 2 ) # works but needs too much gpu mem
    return val ## return final prob


class ProtEncoderMaxpoolConv1d (nn.Module): ## !! follow same approach as https://github.com/bio-ontology-research-group/deepgo/blob/master/nn_hierarchical_seq.py#L148
  def __init__(self,args,**kwargs):
    super().__init__()
    self.args = args

    ## takes kmer as input
    self.KmerEmb = nn.Embedding(args.num_kmer,args.kmer_dim,padding_idx=0) ## must have padding index=0
    xavier_uniform_(self.KmerEmb.weight) ## good init value ?

    self.dropout = nn.Dropout(kwargs['dropout'])

    ## DeepGO input is 128, 32, len=128, so this is a square block 128x128 conv. do this 32 times.
    self.conv1d = nn.Conv1d (args.kmer_dim, args.kmer_dim_out, kernel_size=args.kmer_filter_size, stride=1, padding=0) ## keras code uses MaxPooling1D padding='valid' as no padding https://keras.io/layers/convolutional/
    xavier_uniform_(self.conv1d.weight) ## good init value ?

    self.maxpool1d = nn.MaxPool1d( kernel_size=64, stride=32, padding=0)

  def forward(self, input_idx, **kwargs):
    ## !! follow same approach as https://github.com/bio-ontology-research-group/deepgo/blob/master/nn_hierarchical_seq.py#L148
    kmer_emb = self.KmerEmb ( input_idx ) ## batch x word x dim
    kmer_emb = self.dropout ( kmer_emb )

    ## conv over the length of the protein
    kmer_emb = F.relu ( self.conv1d (kmer_emb.transpose(1,2)) ) ## in-dim is kmer dim, not kmer length, output here is batch x kmer_emb x max_input_len

    kmer_emb = self.maxpool1d (kmer_emb) ## output should be batch x kmer_emb x new_len https://pytorch.org/docs/stable/nn.html#pooling-layers

    kmer_emb = torch.flatten( kmer_emb, start_dim=1 ) ## 0-padding will be added to end of vector ... this is theoretically incorrect
    ## with the same setting as deepgo, we get batch x 832 (not include ProtNetwork)
    return kmer_emb # kmer_emb is now 2D batch x (kmer_dim * max_pool_len)


class ProtSeq2GOBase (nn.Module):
  ## base class that has Prot encoder
  def __init__(self, ProtEncoder, args, **kwargs):
    super(ProtSeq2GOBase, self).__init__()

    self.args = args

    self.ProtEncoder = ProtEncoder
    if self.args.fix_prot_emb:
      for p in self.ProtEncoder.parameters(): ## save a lot of mem when turn off gradient
        p.requires_grad=False

    self.loss_type = 'BCEWithLogitsLoss'
    self.classify_loss = nn.BCEWithLogitsLoss()
    # self.classify_loss = nn.BCELoss()

  def maxpool_prot_emb (self,prot_idx,mask):
    pass

  def match_prob (self, prot_emb, go_emb ) :
    pass

  def forward( self, prot_idx, mask, prot_interact_emb, label_ids, **kwargs ):
    return 0, 0 ## return probability (can be un-sigmoid) and loss value

  def make_optimizer (self):

    print ('\noptim function : {}'.format(self.args.optim_choice)) ## move in here, so that we can simply change @args.optim_choice and call @make_optimizer again
    # if self.args.optim_choice == 'SGD':
    self.optim_choice = torch.optim.SGD
    if self.args.optim_choice == 'Adam':
      self.optim_choice = torch.optim.Adam
    if self.args.optim_choice == 'RMSprop':
      self.optim_choice = torch.optim.RMSprop

    if self.args.fix_prot_emb:
      param_list = [(n,p) for n,p in self.named_parameters() if "ProtEncoder" not in n]

    else:
      param_list = [(n,p) for n,p in self.named_parameters()]

    ## save GPU mem by manually turn off gradients of stuffs we don't optim
    param_name_to_optim = [ n[0] for n in param_list ] ## just get name
    for n,p in self.named_parameters():
      if n not in param_name_to_optim:
        p.requires_grad = False

    print ("\n\nparams to train")
    for n,p in self.named_parameters():
      if p.requires_grad : print (n)

    return self.optim_choice ( [p for n,p in param_list], lr=self.args.lr ) # , momentum=0.9

  def do_train(self, prot_loader, prot_dev_loader, **kwargs):
    torch.cuda.empty_cache()

    optimizer = self.make_optimizer()

    eval_acc = 0
    lowest_dev_loss = np.inf

    for epoch in range( int(self.args.epoch)) :

      self.train()
      tr_loss = 0

      for step, batch in enumerate(tqdm(prot_loader, desc="epoch {}".format(epoch))):

        batch = tuple(t for t in batch) # all_input_ids, all_input_len, all_input_mask, all_label_ids

        with torch.no_grad():
          if self.args.has_ppi_emb:
            prot_idx , prot_len , mask , label_ids , prot_interact_emb = batch ## @label_ids must be of size @args.num_label_to_test
          else:
            prot_idx , prot_len , mask , label_ids , _ = batch

        prot_idx = prot_idx[ :, 0:int(max(prot_len))] ## trim down
        mask = mask[ :, 0:int(max(prot_len))]

        if self.args.has_ppi_emb:
          prot_interact_emb = prot_interact_emb.cuda()
        else:
          prot_interact_emb = None

        pred, loss = self.forward(prot_idx.cuda(), mask.cuda(), prot_interact_emb, label_ids.cuda(), **kwargs)

        # loss = self.classify_loss ( pred, label_ids.cuda() )
        loss.backward() ## we will later define what is @loss, so for base class, we will see error.
        # torch.nn.utils.clip_grad_norm_(self.parameters(), args.max_grad_norm)

        optimizer.step()
        optimizer.zero_grad()

        tr_loss = tr_loss + loss

      ## end epoch
      print ("\ntrain epoch {} loss {}".format(epoch,tr_loss))

      # eval at each epoch
      # print ('\neval on train data epoch {}'.format(epoch)) ## if too slow, comment out. skip for now
      # result, _ , _ = self.do_eval(prot_loader,**kwargs)

      print ('\neval on dev data epoch {}'.format(epoch))
      result, preds, dev_loss = self.do_eval(prot_dev_loader,**kwargs)

      if (dev_loss < lowest_dev_loss) :
        lowest_dev_loss = dev_loss
        print ("save best, lowest dev loss {}".format(lowest_dev_loss))
        torch.save(self.state_dict(), os.path.join(self.args.result_folder,"best_state_dict.pytorch"))
        last_best_epoch = epoch

      else:
        if (epoch > 4) : ## don't decrease too quickly and too early, wait for later epoch
          for g in optimizer.param_groups:
            g['lr'] = g['lr'] * 0.8 ## update lr rate for next epoch

      if epoch - last_best_epoch > 10:
        print ('\n\n\n**** break early \n\n\n')
        print ("save last")
        torch.save(self.state_dict(), os.path.join(self.args.result_folder,"last_state_dict.pytorch"))
        return tr_loss

      elif self.args.switch_sgd and (epoch - last_best_epoch > 4) and (self.args.optim_choice != 'SGD') :
        ## SGD seems to always able to decrease DevSet loss. it is slow, so we don't start with SGD, we will use RMSprop then do SGD
        print ('\n\nload back best state_dict\n\n')
        self.load_state_dict( torch.load ( os.path.join(self.args.result_folder,"best_state_dict.pytorch") ) , strict=False )

        print ('\n\nchange from {} to SGD\n\n'.format(self.args.optim_choice) )
        self.args.optim_choice = 'SGD'
        optimizer = self.make_optimizer()

    print ("save last")
    torch.save(self.state_dict(), os.path.join(self.args.result_folder,"last_state_dict.pytorch"))
    return tr_loss ## last train loss

  def do_eval(self, prot_loader, **kwargs):

    torch.cuda.empty_cache()
    self.eval()

    tr_loss = 0
    preds = []
    all_label_ids = []

    for step, batch in enumerate(prot_loader):

      with torch.no_grad(): ## no gradient for everything in this section

        batch = tuple(t for t in batch)
        if self.args.has_ppi_emb:
          prot_idx , prot_len , mask , label_ids , prot_interact_emb = batch ## @label_ids must be of size @args.num_label_to_test
        else:
          prot_idx , prot_len , mask , label_ids , _ = batch

        prot_idx = prot_idx[ :, 0:int(max(prot_len))] ## trim down
        mask = mask[ :, 0:int(max(prot_len))]

        if self.args.has_ppi_emb:
          prot_interact_emb = prot_interact_emb.cuda()
        else:
          prot_interact_emb = None

        pred, loss = self.forward(prot_idx.cuda(), mask.cuda(), prot_interact_emb, label_ids.cuda(), **kwargs)

        # loss = self.classify_loss ( pred, label_ids.cuda() )

      tr_loss = tr_loss + loss

      ## take sgimoid here, if sigmoid was not taken inside @forward
      if self.loss_type == 'BCEWithLogitsLoss':
        pred = F.sigmoid(pred)

      if len(preds) == 0:
        preds.append(pred.detach().cpu().numpy())
        all_label_ids.append(label_ids.detach().cpu().numpy())
      else:
        preds[0] = np.append(preds[0], pred.detach().cpu().numpy(), axis=0)
        all_label_ids[0] = np.append(all_label_ids[0], label_ids.detach().cpu().numpy(), axis=0) # row array

    # end eval
    all_label_ids = all_label_ids[0]
    preds = preds[0]

    print ('loss {}'.format(tr_loss))

    print ('pred label')
    print (preds)

    print ('true label')
    print (all_label_ids)

    trackF1macro = {}
    trackF1micro = {} # metrics["f1_micro"]

    trackMacroPrecision = {} # [MACRO] accuracy, precision, recall
    trackMacroRecall = {}

    trackMicroPrecision = {}
    trackMicroRecall = {}

    ## DO NOT NEED TO DO THIS ALL THE TIME DURING TRAINING
    if self.args.not_train:
      rounding = np.arange(.1,1,.4)
    else:
      rounding = [0.5]

    for round_cutoff in rounding:

      print ('\n\nround cutoff {}'.format(round_cutoff))

      preds_round = 1.0*( round_cutoff < preds ) ## converted into 0/1

      result = evaluation_metric.all_metrics ( preds_round , all_label_ids, yhat_raw=preds, k=[5,10,15,20,25]) ## we can pass vector of P@k and R@k
      evaluation_metric.print_metrics( result )

      if 'full_data' not in trackF1macro:
        trackF1macro['full_data'] = [result["f1_macro"]]
        trackF1micro['full_data'] = [result["f1_micro"]]
        trackMacroPrecision['full_data'] = [result["prec_macro"]]
        trackMicroPrecision['full_data'] = [result["prec_micro"]]
        trackMacroRecall['full_data'] = [result["rec_macro"]]
        trackMicroRecall['full_data'] = [result["rec_micro"]]
      else:
        trackF1macro['full_data'].append(result["f1_macro"])
        trackF1micro['full_data'].append(result["f1_micro"])
        trackMacroPrecision['full_data'].append(result["prec_macro"])
        trackMicroPrecision['full_data'].append(result["prec_micro"])
        trackMacroRecall['full_data'].append(result["rec_macro"])
        trackMicroRecall['full_data'].append(result["rec_micro"])

      if ('GoCount' in kwargs ) and (self.args.not_train): ## do not eed to do this all the time 
        print ('\n\nsee if method improves accuracy conditioned on frequency of GO terms')

        ## frequency less than 25 quantile  and over 75 quantile
        ## indexing must be computed ahead of time to to avoid redundant calculation

        for cutoff in ['quant25','quant75','betweenQ25Q75']:
          ## indexing of the column to pull out , @pred is num_prot x num_go
          result = evaluation_metric.all_metrics ( preds_round[: , kwargs[cutoff]] , all_label_ids[: , kwargs[cutoff]], yhat_raw=preds[: , kwargs[cutoff]], k=[5,10,15,20,25])
          print ("\nless than {} count".format(cutoff))
          evaluation_metric.print_metrics( result )

          if cutoff not in trackF1macro:
            trackF1macro[cutoff] = [result["f1_macro"]]
            trackF1micro[cutoff] = [result["f1_micro"]]
            trackMacroPrecision[cutoff] = [result["prec_macro"]]
            trackMicroPrecision[cutoff] = [result["prec_micro"]]
            trackMacroRecall[cutoff] = [result["rec_macro"]]
            trackMicroRecall[cutoff] = [result["rec_micro"]]
          else:
            trackF1macro[cutoff].append(result["f1_macro"])
            trackF1micro[cutoff].append(result["f1_micro"])
            trackMacroPrecision[cutoff].append(result["prec_macro"])
            trackMicroPrecision[cutoff].append(result["prec_micro"])
            trackMacroRecall[cutoff].append(result["rec_macro"])
            trackMicroRecall[cutoff].append(result["rec_micro"])


    ##
    if self.args.not_train :
      print ('\n\ntracking f1 compile into list\n')

      # print ('\nmacro f1 prec rec')
      for k,v in trackF1macro.items():
        print ('macroF1 ' + k + " " + " ".join(str(s) for s in v))

      for k,v in trackMacroPrecision.items():
        print ('macroPrec ' + k + " " + " ".join(str(s) for s in v))

      for k,v in trackMacroRecall.items():
        print ('macroRec ' + k + " " + " ".join(str(s) for s in v))

      # print ('\nmicro f1 prec rec')
      for k,v in trackF1micro.items():
        print ('microF1 ' + k + " " + " ".join(str(s) for s in v))

      for k,v in trackMicroPrecision.items():
        print ('microPrec ' + k + " " + " ".join(str(s) for s in v))

      for k,v in trackMicroRecall.items():
        print ('microRec ' + k + " " + " ".join(str(s) for s in v))

    return result, preds, tr_loss


class DeepGOFlatSeqOnly (ProtSeq2GOBase):
  ## run 1D or 2D conv then use attention layer
  def __init__(self,ProtEncoder, args, **kwargs):
    super().__init__(ProtEncoder, args, **kwargs)

    ## do not care about GO vectors, so only need to change forward to not take GO vectors
    self.LinearRegression = nn.Linear( args.go_vec_dim + args.prot_vec_dim, args.num_label_to_test ) ## use trick
    xavier_uniform_(self.LinearRegression.weight)

  def match_prob (self, prot_emb ) :
    ## from deepgo Keras, we have single Dense(1) for each GO term. This is the same as linear projection
    # prot_emb is batch x go_vec_dim
    # prot_emb.unsqueeze(1) is batch x 1 x go_vec_dim , so that we can do broadcast pointwise multiplication
    # self.LinearRegression.bias is 1D array

    pred = self.LinearRegression.weight.mul(prot_emb.unsqueeze(1)).sum(2) + self.LinearRegression.bias ## dot-product sum up
    return pred

  def forward( self, prot_idx, mask, prot_interact_emb, label_ids, **kwargs ): ## @go_emb should be joint train?
    prot_emb = self.ProtEncoder ( prot_idx, **kwargs )
    pred = self.match_prob ( prot_emb )
    loss = self.classify_loss ( pred, label_ids.cuda() )
    return pred, loss


class DeepGOFlatSeqProt (DeepGOFlatSeqOnly):
  ## run 1D or 2D conv then use attention layer
  def __init__(self,ProtEncoder, args, **kwargs):
    super().__init__(ProtEncoder, args, **kwargs)
    self.LinearRegression = nn.Linear( args.go_vec_dim + args.prot_vec_dim + args.prot_interact_vec_dim, args.num_label_to_test ) ## 3 items, go vec, prot vec by some encoder, and prot vec made by some interaction network
    xavier_uniform_(self.LinearRegression.weight)

  def forward( self, prot_idx, mask, prot_interact_emb, label_ids, **kwargs ):
    prot_emb = self.ProtEncoder ( prot_idx, **kwargs )
    ## append the prot_emb by kmer (or some other method) with its vector from a prot-prot interaction network
    prot_emb = torch.cat ( (prot_emb , prot_interact_emb) , dim=1 ) ## out put is 2D, batch x ( prot_emb_dim + prot_interact_emb_dim )
    pred = self.match_prob ( prot_emb )
    loss = self.classify_loss ( pred, label_ids.cuda() )
    return pred, loss


class DeepGOTreeSeqOnly (DeepGOFlatSeqOnly):
  def __init__(self,ProtEncoder, args, **kwargs):
    super().__init__(ProtEncoder, args, **kwargs)
    ## do not care about GO vectors, so only need to change forward to not take GO vectors
    self.maxpoolNodeLayer = maxpoolNodeLayer (kwargs['AdjacencyMatrix'][kwargs['label_to_test_index']])
    self.classify_loss = nn.BCELoss() ## return sigmoid to avoid max with 0, so we take traditional BCE without built in logit
    self.loss_type = 'BCE'

  def forward( self, prot_idx, mask, prot_interact_emb, label_ids, **kwargs ): ## @go_emb should be joint train?
    prot_emb = self.ProtEncoder ( prot_idx, **kwargs )
    pred = self.match_prob ( prot_emb )
    pred = F.sigmoid(pred) ## size batch x all_terms_in_ontology because for 1 node, we need to get all its children. we can do careful selection later to save gpu memmory
    pred = self.maxpoolNodeLayer.forward ( pred )
    loss = self.classify_loss ( pred, label_ids.cuda() )
    return pred, loss


class DeepGOTreeSeqProt (DeepGOTreeSeqOnly):
  def __init__(self,ProtEncoder, args, **kwargs):

    super().__init__(ProtEncoder, args, **kwargs)

    self.LinearRegression = nn.Linear( args.go_vec_dim + args.prot_vec_dim + args.prot_interact_vec_dim, args.num_label_to_test ) ## 3 items, go vec, prot vec by some encoder, and prot vec made by some interaction network
    xavier_uniform_(self.LinearRegression.weight)

  def forward( self, prot_idx, mask, prot_interact_emb, label_ids, **kwargs ): ## @go_emb should be joint train?
    prot_emb = self.ProtEncoder ( prot_idx, **kwargs )
    prot_emb = torch.cat ( (prot_emb , prot_interact_emb) , dim=1 )
    pred = self.match_prob ( prot_emb )
    pred = F.sigmoid(pred) ## size batch x all_terms_in_ontology because for 1 node, we need to get all its children. we can do careful selection later to save gpu memmory
    pred = self.maxpoolNodeLayer.forward ( pred )
    loss = self.classify_loss ( pred, label_ids.cuda() )
    return pred, loss


class DeepGOFlatSeqProtHwayGo (DeepGOFlatSeqProt):
  ## run 1D or 2D conv then use attention layer
  def __init__(self,ProtEncoder, GOEncoder, args, **kwargs):
    super().__init__(ProtEncoder, args)
    self.GOEncoder = GOEncoder
    if self.args.fix_go_emb:
      for p in self.GOEncoder.parameters():
        p.requires_grad=False

    ## prot encoder (from deepgo has dim ~1088)
    ## transform only original prot vector (not adding in ppi-vector)
    ## make into same dim as go vector
    # self.ReduceProtEmb = nn.Linear(args.prot_vec_dim, args.go_vec_dim)
    # xavier_uniform_(self.ReduceProtEmb.weight)

    dim_in = args.prot_vec_dim + args.prot_interact_vec_dim + args.go_vec_dim
    dim_out = args.prot_vec_dim + args.prot_interact_vec_dim
    self.ReduceProtGoEmb = nn.Sequential(nn.Linear(dim_in, dim_out),
                                         nn.ReLU())

    xavier_uniform_(self.ReduceProtGoEmb[0].weight)
    # xavier_uniform_(self.ReduceProtGoEmb[3].weight)

    ## notice we change dim of this Linear Layer
    self.LinearRegression = nn.Linear( dim_out*2 , args.num_label_to_test )
    xavier_uniform_(self.LinearRegression.weight)

  def make_optimizer (self):

    print ('\noptim function : {}'.format(self.args.optim_choice)) ## move in here, so that we can simply change @args.optim_choice and call @make_optimizer again
    # if self.args.optim_choice == 'SGD':
    self.optim_choice = torch.optim.SGD
    if self.args.optim_choice == 'Adam':
      self.optim_choice = torch.optim.Adam
    if self.args.optim_choice == 'RMSprop':
      self.optim_choice = torch.optim.RMSprop


    if self.args.fix_prot_emb and (not self.args.fix_go_emb):
      param_list = [(n,p) for n,p in self.named_parameters() if "ProtEncoder" not in n]

    elif self.args.fix_prot_emb and self.args.fix_go_emb:
      param_list = [(n,p) for n,p in self.named_parameters() if ("ProtEncoder" not in n) or ("GOEncoder" not in n)]

    elif (not self.args.fix_prot_emb) and self.args.fix_go_emb:
      param_list = [(n,p) for n,p in self.named_parameters() if "GOEncoder" not in n]

    else:
      param_list = [(n,p) for n,p in self.named_parameters()]

    ## save GPU mem by manually turn off gradients of stuffs we don't optim
    param_name_to_optim = [ n[0] for n in param_list ] ## just get name
    for n,p in self.named_parameters():
      if n not in param_name_to_optim:
        p.requires_grad = False

    print ("\n\nparams to train")
    for n,p in self.named_parameters():
      if p.requires_grad : print (n)

    return self.optim_choice ( [p for n,p in param_list], lr=self.args.lr ) # , momentum=0.9

  def concat_prot_go (self, prot_emb, go_emb):

    ## should we do one single model for all GO terms ? or each different model for each GO term ?

    ## reduce sparse prot vector into dense space ~1000 --> 300
    ## OR .... we expland go_emb 300 --> ~1000

    prot_emb = prot_emb.unsqueeze(1) ## make into 3D : batch x 1 x dim
    prot_emb = prot_emb.expand(-1, self.args.num_label_to_test, -1 ) ## batch x num_go x dim ... -1 implies default

    ## only need unsqueeze if we change the 2nd or 3rd dim
    go_emb = go_emb.expand(prot_emb.shape[0], -1, -1 ) ## the -1 implies default dim. num_prot x num_go x dim (so we duplicated same matrix based on #prot

    prot_go_vec = torch.cat((prot_emb,go_emb), dim=2) # append prot_emb with go_emb. prot_go_vec[0] is 1st protein in the batch concat with many go terms (its num. of row)

    prot_go_vec = self.ReduceProtGoEmb (prot_go_vec) ## somehow make interaction between prot and go

    prot_go_vec = torch.cat ( (prot_emb , prot_go_vec) , dim=2 ) ## append to f(prot_emb,go_emb)

    return prot_go_vec ## output shape batch x num_go x dim

  def forward( self, prot_idx, mask, prot_interact_emb, label_ids, **kwargs ):
    ## @go_emb should be joint train?
    # go_emb will be computed each time. so it's costly ?

    ## Because of GCN, GO vectors not in subset will influence GO vectors inside subset, so we have to run GCN on the whole graph ? ... probably yes
    ## **** TO SAVE SOME TIME, IF WE DON'T UPDATE LABEL DESCRIPTION, WE CAN COMPUTE IT AHEAD OF TIME.
    # if self.args.fix_go_emb:
    go_emb = kwargs['go_emb']

    # else:
    #   go_emb = self.GOEncoder.gcn_2layer(kwargs['labeldesc_loader'],kwargs['edge_index']) ## go_emb is num_go x dim
    #   go_emb = F.normalize(go_emb,dim=1) ## go emb were trained based on cosine, so we have norm to length=1, this normalization will get similar GO to have truly similar vectors.

    # prot_emb is usually fast, because we don't update it ? must update if we use deepgo approach
    prot_emb = self.ProtEncoder ( prot_idx, **kwargs ) ## output is batch x dim if we use deepgo. possible we need to change later.

    ## append @prot_emb to vector from prot-prot interaction network
    prot_emb = torch.cat ( (prot_emb, prot_interact_emb) , dim=1 ) ## append to known ppi-network

    ## somehow ... account for the GO vectors
    ## highway network for @prot_emb_concat

    ## testing on subset, so we call kwargs['label_to_test_index']
    ## if we don't care about using graph/ or children terms, then @label_to_test_index should be just simple index 0,1,2,3,....
    prot_go_vec = self.concat_prot_go ( prot_emb, go_emb[kwargs['label_to_test_index']] ) ## output shape batch x num_go x dim

    ## testing on subset
    pred = self.LinearRegression.weight.mul(prot_go_vec).sum(2) + self.LinearRegression.bias ## dot-product sum up

    loss = self.classify_loss ( pred, label_ids.cuda() )

    return pred, loss


class DeepGOFlatSeqProtHwayNotUseGo (DeepGOFlatSeqProtHwayGo):

  def __init__(self,ProtEncoder, GOEncoder, args, **kwargs):

    ## this object is meant to show that using the extra highway network without GO vector DOES NOT IMPROVE RESULTS

    super().__init__(ProtEncoder, GOEncoder, args, **kwargs)

    dim_in = args.prot_vec_dim + args.prot_interact_vec_dim ## INSTEAD OF HAVING GO VEC, WE HAVE EXTRA @prot_vec_dim
    dim_out = args.prot_vec_dim + args.prot_interact_vec_dim
    self.ReduceProtGoEmb = nn.Sequential(nn.Linear(dim_in, dim_out),
                                         nn.ReLU())

    xavier_uniform_(self.ReduceProtGoEmb[0].weight)
    # xavier_uniform_(self.ReduceProtGoEmb[3].weight)

    ## notice we change dim of this Linear Layer
    self.LinearRegression = nn.Linear( dim_out*2 , args.num_label_to_test )
    xavier_uniform_(self.LinearRegression.weight)

  def forward( self, prot_idx, mask, prot_interact_emb, label_ids, **kwargs ):

    # prot_emb is usually fast, because we don't update it ? must update if we use deepgo approach
    prot_emb = self.ProtEncoder ( prot_idx, **kwargs ) ## output is batch x dim if we use deepgo. possible we need to change later.

    ## append @prot_emb to vector from prot-prot interaction network
    prot_emb = torch.cat ( (prot_emb, prot_interact_emb) , dim=1 ) ## append to known ppi-network

    ## CREATE INTERACTION TERM BUT DO NOT USE GO VECTOR
    prot_emb_interaction_term = self.ReduceProtGoEmb (prot_emb)

    prot_emb = torch.cat ( (prot_emb, prot_emb_interaction_term), dim=1 ) ## this is num_batch x dim

    ## testing on subset
    pred = self.LinearRegression.weight.mul(prot_emb.unsqueeze(1)).sum(2) + self.LinearRegression.bias ## dot-product sum up
    loss = self.classify_loss ( pred, label_ids.cuda() )

    return pred, loss


class DeepGOFlatSeqProtConcatGo (DeepGOFlatSeqProt):
  def __init__(self,ProtEncoder, GOEncoder, args, **kwargs):
    super().__init__(ProtEncoder, args)
    self.GOEncoder = GOEncoder
    if self.args.fix_go_emb:
      for p in self.GOEncoder.parameters():
        p.requires_grad=False

    dim_in = 4*(args.prot_vec_dim + args.prot_interact_vec_dim) ## times 4 because we have 4 vectors as input (see function @ConcatCompare)
    self.CompareMetric = ConcatCompare(dim_in,1,**kwargs) ## predict yes/no, so output is dim 1

  def forward (self, prot_idx, mask, prot_interact_emb, label_ids, **kwargs):

    ## Because of GCN, GO vectors not in subset will influence GO vectors inside subset, so we have to run GCN on the whole graph ? ... probably yes
    go_emb = self.GOEncoder.gcn_2layer(kwargs['labeldesc_loader'],kwargs['edge_index']) ## go_emb is num_go x dim
    go_emb = F.normalize(go_emb,dim=1) ## go emb were trained based on cosine, so we have norm to length=1, this normalization will get similar GO to have truly similar vectors.

    # prot_emb is usually fast, because we don't update it ? must update if we use deepgo approach
    prot_emb = self.ProtEncoder ( prot_idx, **kwargs ) ## output is batch x dim if we use deepgo. possible we need to change later.
    ## append @prot_emb to vector from prot-prot interaction network
    prot_emb = torch.cat ( (prot_emb, prot_interact_emb) , dim=1 ) ## append to known ppi-network

    prot_emb = prot_emb.unsqueeze(1) ## make into 3D : batch x 1 x dim
    prot_emb = prot_emb.expand( -1, self.args.num_label_to_test, -1) ## output shape batch x num_go x dim

    ## only need unsqueeze if we change the 2nd or 3rd dim
    go_emb_select = go_emb[kwargs['label_to_test_index']] ## testing on subset
    go_emb_select = go_emb_select.expand(prot_idx.shape[0], -1, -1 ) ## the -1 implies default dim. num_prot x num_go x dim (so we duplicated same matrix based on #prot

    pred = self.CompareMetric(prot_emb,go_emb_select) ## output num_prot x num_go
    loss = self.classify_loss ( pred, label_ids.cuda() )

    return pred, loss


class DeepGOTreeSeqProtHwayGo (DeepGOFlatSeqProtHwayGo):

  def __init__(self,ProtEncoder, GOEncoder, args, **kwargs):
    super().__init__(ProtEncoder, GOEncoder, args, **kwargs)

    self.classify_loss = nn.BCELoss() ## return sigmoid to avoid max with 0, so we take traditional BCE without built in logit
    self.loss_type = 'BCE'

    self.maxpoolNodeLayer = maxpoolNodeLayer (kwargs['AdjacencyMatrix'][kwargs['label_to_test_index']]) # use @label_to_test_index because we need to extract children of nodes we're testing on

  def forward( self, prot_idx, mask, prot_interact_emb, label_ids, **kwargs ):
    ## @go_emb should be joint train?
    # go_emb will be computed each time. so it's costly ?

    ## **** TO SAVE SOME TIME, IF WE DON'T UPDATE LABEL DESCRIPTION, WE CAN COMPUTE IT AHEAD OF TIME.
    # if self.args.fix_go_emb:
    go_emb = kwargs['go_emb']
    # else:
    #   ## Because of GCN, GO vectors not in subset will influence GO vectors inside subset, so we have to run GCN on the whole graph ? ... probably yes
    #   go_emb = self.GOEncoder.gcn_2layer(kwargs['labeldesc_loader'],kwargs['edge_index']) ## go_emb is num_go x dim
    #   go_emb = F.normalize(go_emb,dim=1) ## go emb were trained based on cosine, so we have norm to length=1, this normalization will get similar GO to have truly similar vectors.

    # prot_emb is usually fast, because we don't update it ? must update if we use deepgo approach
    prot_emb = self.ProtEncoder ( prot_idx, **kwargs ) ## output is batch x dim if we use deepgo. possible we need to change later.

    ## append @prot_emb to vector from prot-prot interaction network
    prot_emb = torch.cat ( (prot_emb, prot_interact_emb) , dim=1 ) ## append to known ppi-network

    ## somehow ... account for the GO vectors
    ## testing on subset, so we call kwargs['label_to_test_index']
    prot_go_vec = self.concat_prot_go ( prot_emb, go_emb[kwargs['label_in_ontology_index']] ) ## output shape batch x num_go x dim

    ## testing on subset
    pred = self.LinearRegression.weight.mul(prot_go_vec).sum(2) + self.LinearRegression.bias ## dot-product sum up

    pred = F.sigmoid(pred) ## size batch x all_terms_in_ontology because for 1 node, we need to get all its children. we can do careful selection later to save gpu memmory

    pred = self.maxpoolNodeLayer.forward ( pred ) ## max over all GO terms

    loss = self.classify_loss ( pred, label_ids.cuda() ) ## get accuracy only for labels we care for.

    return pred, loss


# class ProtSeq2GOBert (ProtSeq2GOGcn):
#   ## run 1D or 2D conv then use attention layer
#   def __init__(self,ProtEncoder, GOEncoder, args):
#     super().__init__(ProtEncoder, GOEncoder, args)

#   def bert_go_enc (self, label_desc_loader):
#     # must send go description in batch mode ... slow ?
#     # backprop on bert will be slow too ?

#     go_emb = torch.zeros(self.args.num_label_to_test, self.args.go_vec_dim)
#     start = 0

#     for step, batch in enumerate(label_desc_loader):

#       batch = tuple(t for t in batch)
#       label_desc, label_len, label_mask = batch

#       label_desc.data = label_desc.data[ : , 0:int(max(label_len)) ] # trim down input to max len of the batch
#       label_mask.data = label_mask.data[ : , 0:int(max(label_len)) ] # trim down input to max len of the batch
#       label_emb = self.GOEncoder.encode_label_desc(label_desc,label_len,label_mask)
#       if self.args.reduce_cls_vec: ## can use 768 output, or linear reduce to 300
#         label_emb = self.GOEncoder.metric_module.reduce_vec_dim(label_emb)

#       end = start+label_desc.shape[0]
#       go_emb[start:end] = label_emb

#       start = end ## next position

#     return go_emb

#   def forward( self, prot_idx, mask, **kwargs ):
#     ## @go_emb should be joint train?
#     # go_emb will be computed each time. so it's costly ?
#     go_emb = self.GOEncoder.bert_go_enc(kwargs['labeldesc_loader'])

#     # prot_emb is usually fast, because we don't update it ?
#     prot_emb = self.maxpool_prot_emb ( prot_idx, mask )
#     pred = self.match_prob ( prot_emb, go_emb )
#     return pred







