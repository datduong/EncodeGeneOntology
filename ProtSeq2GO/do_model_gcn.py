

from __future__ import absolute_import, division, print_function

import argparse,csv,logging,os,random,sys,pickle,gzip,json,re
import numpy as np
import pandas as pd
from tqdm import tqdm, trange


import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.init import xavier_uniform_

from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from torch.nn import CrossEntropyLoss, MSELoss
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score

from pytorch_pretrained_bert.tokenization import BertTokenizer, load_vocab

from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

# print (sys.path)

sys.path.append("/local/datdb/GOmultitask")

import GCN.encoder.data_loader as GCN_data_loader
import GCN.encoder.encoder_model as GCN_encoder_model

os.chdir("/local/datdb/GOmultitask")

import ProtSeq2GO.ProtSeq2GOModel as ProtSeq2GOModel
import ProtSeq2GO.protSeqLoader as protSeqLoader

import ProtSeq2GO.arg_input as arg_input
args = arg_input.get_args()

print (args)

MAX_SEQ_LEN = 2001

os.chdir(args.main_dir)


## **** load edges ****

edge_index = pickle.load ( open("adjacency_matrix_coo_format.pickle","rb") )
edge_index = torch.tensor( edge_index, dtype=torch.long)

print ('num of edges {}'.format(edge_index.shape))


all_name_array = pd.read_csv("go_name_in_obo.csv", header=None)
all_name_array = list (all_name_array[0])
args.num_label = len(all_name_array)

## **** supposed we want to do testing on subset, to repeat deepgo baseline
label_to_test = None
label_to_test_index = None

if (args.num_label_to_test == 0) and (args.label_subset_file is None):
  args.num_label_to_test = args.num_label

elif args.label_subset_file is not None: ## we must properly extract the labels wanted

  print ('\n\nloading this set of go terms {}'.format(args.label_subset_file))

  ## we can do legacy GO terms if we don't care about the definitions

  label_to_test = pd.read_csv(args.label_subset_file, header=None)
  label_to_test = sorted ( list (label_to_test[0]) ) ## must sort, because we sort @all_name_array. we want the same ordering if we use @edge_index and run GCN to get emb. ORDERING DOES NOT MATTER IF WE DON'T USE GO DEFINITIONS

  print ('\n\ntotal label to test before filter legacy terms {}'.format(len(label_to_test)))

  label_to_test = [l for l in label_to_test if l in all_name_array] ## legacy terms will not be found in newest data

  args.num_label_to_test = len(label_to_test)
  print ('total label to test {}\n\n'.format(len(label_to_test)))

  ## get indexing of label to test from @all_name_array
  ## **** must be using the indexing from @all_name_array because GCN requires the graph, and @edge_index is made from this @all_name_array
  ## **** BERT will not need this graph, so for bert, maybe we just replace @all_name_array with @label_to_test ?

  label_to_test_index = np.array ( [all_name_array.index(label) for label in label_to_test] )
  # label_to_test_index = np.arange( args.num_label_to_test ) ## extract all labels

  # print ('\n\nlabel_to_test_index')
  # print (label_to_test_index)
  # print ('\n\n')



## **** create maxpool just like how deepgo does it. ?? cannot do that right now. too slow, takes too much mem
AdjacencyMatrix = None
label_in_ontology_index = None
if args.tree :
  # create label_in_ontology_index
  label_in_ontology = pd.read_csv(args.label_in_ontology, header=None)
  label_in_ontology = sorted (list ( label_in_ontology[0] ) )
  label_in_ontology_index = np.array ( [all_name_array.index(label) for label in label_in_ontology] ) ## used to extract terms in MF or BP or CC from a large @go_emb

  args.num_label_to_test = len(label_in_ontology)
  print ('because using tree, we need to add children terms, update total label to test {}\n\n'.format(len(label_in_ontology)))

  ## redo @label_to_test_index so that it matches the indexing of @label_in_ontology
  ## conditioned on @label_in_ontology, what is the index of each label to test. 
  label_to_test_index = np.array ( [label_in_ontology.index(label) for label in label_to_test] )

  AdjacencyMatrix = pickle.load ( gzip.open(args.adjacency_matrix+".gzip.pickle","rb") )
  AdjacencyMatrix = torch.FloatTensor(AdjacencyMatrix)


if args.use_cuda:
  edge_index = edge_index.cuda()


## **** load protein data

# all_name_array = [ re.sub(r"GO:","",g) for g in all_name_array ] ## don't use GO: in the input files

if args.ontology is None:
  add_name = ""
else:
  add_name = '-' + args.ontology

train_loader = protSeqLoader.ProtLoader (args.data_dir, 'train'+add_name+'.tsv', all_name_array, MAX_SEQ_LEN, 'random', args, args.do_kmer, label_to_test)

dev_loader = protSeqLoader.ProtLoader (args.data_dir, 'dev'+add_name+'.tsv', all_name_array, MAX_SEQ_LEN, 'sequential', args, args.do_kmer, label_to_test)


## **** make model ****

nonlinear_gcnn = {'tanh':F.tanh, 'relu':F.relu}

other_params = {'dropout': 0.2,
                'metric_option': args.metric_option,
                'nonlinear_gcnn': nonlinear_gcnn[args.nonlinear_gcnn],
                'labeldesc_loader': None,
                'edge_index': edge_index,
                'AdjacencyMatrix': AdjacencyMatrix,
                'label_to_test_index': label_to_test_index,
                'label_in_ontology_index': label_in_ontology_index
                }

pretrained_weight = None
if args.w2v_emb is not None:
  pretrained_weight = pickle.load(open(args.w2v_emb,'rb'))

  try: ## load standard pickle that is already in numpy
    pretrained_weight.shape[0] ## will throw error if load in file is not a matrix
  except: ## onto2vec is dictionary {go:vec}
    temp = np.zeros((len(all_name_array), args.def_emb_dim ))
    for index,go in enumerate (all_name_array):  ## must keep this exact order
      if go not in pretrained_weight: 
        go = re.sub("GO:","",go)
      # enforce strict "GO:xyz" but onto2vec doesn't have this
      temp[index] = pretrained_weight[go]

    ## now we get word dim and so forth
    pretrained_weight = temp ## override
    other_params ['num_of_word'] = pretrained_weight.shape[0]
    other_params ['word_vec_dim'] = pretrained_weight.shape[1]
    other_params ['pretrained_weight'] = pretrained_weight



## **** load GO count dictionary data
if args.label_counter_dict is not None:
  GO_counter = pickle.load(open(args.label_counter_dict,"rb"))
  quant25, quant75 = protSeqLoader.GetCountQuantile(GO_counter)
  betweenQ25Q75 = protSeqLoader.IndexBetweenQ25Q75Quantile(label_to_test,GO_counter,quant25,quant75)
  quant25 = protSeqLoader.IndexLessThanQuantile(label_to_test,GO_counter,quant25)
  quant75 = protSeqLoader.IndexMoreThanQuantile(label_to_test,GO_counter,quant75)
  
  print ('counter 25 and 75 quantiles {} {}'.format(quant25, quant75))

  other_params['GoCount'] = GO_counter
  other_params['quant25'] = quant25
  other_params['quant75'] = quant75
  other_params['betweenQ25Q75'] = betweenQ25Q75


# cosine model
# **** in using cosine model, we are not using the training sample A->B then B not-> A

cosine_loss = GCN_encoder_model.cosine_distance_loss(args.gcnn_dim,args.gcnn_dim, args)

metric_pass_to_joint_model = {'entailment':None, 'cosine':cosine_loss}

## make GCN model
if args.w2v_emb is None:
  GOEncoder = GCN_encoder_model.encoder_model ( args, metric_pass_to_joint_model[args.metric_option], **other_params )
else: 
  if args.word_mode == 'PretrainedGO':
    GOEncoder = GCN_encoder_model.encoder_model_extended_embedding ( args, metric_pass_to_joint_model[args.metric_option], **other_params )


if args.go_enc_model_load is not None:
  print ('\n\nload back best model for GO encoder {}'.format(args.go_enc_model_load))
  GOEncoder.load_state_dict( torch.load( args.go_enc_model_load ), strict=False )

if args.fix_go_emb and (args.go_vec_dim > 0):
  GOEncoder.cuda()
  with torch.no_grad():
    go_emb = GOEncoder.gcn_2layer(other_params['labeldesc_loader'],other_params['edge_index']) ## go_emb is num_go x dim
    other_params['go_emb'] = F.normalize(go_emb,dim=1)



## load in ProtEncoder
if args.do_kmer:
  # create kmer object
  ProtEncoder = ProtSeq2GOModel.ProtEncoderMaxpoolConv1d(args, **other_params)

else:
  ProtEncoder = torch.load("/local/datdb/ProteinEmbMethodGithub/protein-sequence-embedding-iclr2019/pretrained_models/ssa_L1_100d_lstm3x512_lm_i512_mb64_tau0.5_lambda0.1_p0.05_epoch100.sav")


## make ProtSeq2GO model

if args.model_choice == 'DeepGOTreeSeqOnly': 
  prot2seq_model = ProtSeq2GOModel.DeepGOTreeSeqOnly (ProtEncoder, args, **other_params)

if args.model_choice == 'DeepGOTreeSeqProt': 
  prot2seq_model = ProtSeq2GOModel.DeepGOTreeSeqProt (ProtEncoder, args, **other_params)

if args.model_choice == 'DeepGOFlatSeqProt': 
  prot2seq_model = ProtSeq2GOModel.DeepGOFlatSeqProt (ProtEncoder, args, **other_params)

if args.model_choice == 'DeepGOFlatSeqOnly': 
  prot2seq_model = ProtSeq2GOModel.DeepGOFlatSeqOnly (ProtEncoder, args, **other_params)

if args.model_choice == 'DeepGOFlatSeqProtHwayGo': 
  prot2seq_model = ProtSeq2GOModel.DeepGOFlatSeqProtHwayGo (ProtEncoder, GOEncoder, args, **other_params)

if args.model_choice == 'DeepGOTreeSeqProtHwayGo': 
  prot2seq_model = ProtSeq2GOModel.DeepGOTreeSeqProtHwayGo (ProtEncoder, GOEncoder, args, **other_params)

if args.prot2seq_model_load is not None:
  print ('\n\nload back best model to continue train {}'.format(args.prot2seq_model_load))
  prot2seq_model.load_state_dict( torch.load( args.prot2seq_model_load ), strict=False )


print ('\n\nsee model')
print (prot2seq_model)

prot2seq_model.cuda()

##
# random.seed(args.seed)
# np.random.seed(args.seed)
# torch.manual_seed(args.seed)

## **** train

if not args.not_train: 
  prot2seq_model.do_train(train_loader, dev_loader, **other_params)


## **** run on test set
## load back best model on dev 
print ('\n\nload back best model based on dev set {}'.format(os.path.join(args.result_folder,"best_state_dict.pytorch")))
prot2seq_model.load_state_dict( torch.load( os.path.join(args.result_folder,"best_state_dict.pytorch") ), strict=False )

## load test set 
test_loader = protSeqLoader.ProtLoader (args.data_dir, 'test'+add_name+'.tsv', all_name_array, MAX_SEQ_LEN, 'sequential', args, args.do_kmer, label_to_test)
print ('\non test set\n')
prot2seq_model.do_eval(test_loader, **other_params)

