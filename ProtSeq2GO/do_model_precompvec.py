

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

all_name_array = pd.read_csv("go_name_in_obo.csv", header=None)
all_name_array = list (all_name_array[0])
args.num_label = len(all_name_array)


## **** supposed we want to do testing on subset, to repeat deepgo baseline
label_to_test = None
label_to_test_index = None

if (args.num_label_to_test == 0) and (args.label_subset_file is None):
  args.num_label_to_test = args.num_label


elif args.label_subset_file is not None:

  ## we must properly extract the labels wanted
  ## if we do not want to update GO encoder, then we don't actually need this "if" statement. we can directly remove legacy terms from @go_name_in_obo.csv when creating @all_name_array

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

  # label_to_test_index = np.array ( [all_name_array.index(label) for label in label_to_test] )
  label_to_test_index = np.arange( args.num_label_to_test ) ## extract all labels


## **** create maxpool just like how deepgo does it. ??, then we need to load the entire BP/CC/MF
AdjacencyMatrix = None
label_in_ontology_index = None

if args.tree :

  label_in_ontology = pd.read_csv(args.label_in_ontology, header=None)
  label_in_ontology = sorted (list ( label_in_ontology[0] ) )

  # create label_in_ontology_index
  label_in_ontology_index = np.array ( [all_name_array.index(label) for label in label_in_ontology] ) ## used to extract terms in MF or BP or CC from a large @go_emb

  args.num_label_to_test = len(label_in_ontology)
  print ('because using tree, we need to add children terms, update total label to test {}\n\n'.format(len(label_in_ontology)))

  ## redo @label_to_test_index so that it matches the indexing of @label_in_ontology
  ## conditioned on @label_in_ontology, what is the index of each label to test.
  label_to_test_index = np.array ( [label_in_ontology.index(label) for label in label_to_test] )

  AdjacencyMatrix = pickle.load ( gzip.open(args.adjacency_matrix+".gzip.pickle","rb") )
  AdjacencyMatrix = torch.FloatTensor(AdjacencyMatrix)



## requires labels to be read as "indexing"

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.modeling import BertForMaskedLM, BertConfig, BertForPreTraining
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule

## **** load label description
import biLSTM.encoder.data_loader as biLSTM_data_loader
import biLSTM.encoder.encoder_model as biLSTM_encoder_model
import biLSTM.encoder.entailment_model as biLSTM_entailment_model
import biLSTM.encoder.bi_lstm_model as bi_lstm_model

MAX_SEQ_LEN_LABEL_DEF = 512 ## max len for GO def (probably can be smaller)

if args.w2v_emb is not None: ## we can just treat each node as a vector without word description 
  Vocab = load_vocab(args.vocab_list) # all words found in pubmed and trained in w2v ... should trim down

## reading in feature label is in @GCN folder. too lazy to port this function out.
LabelDescLoader = GCN_data_loader.LabelProcessor()

if args.tree:
  # @label_in_ontology to get GO in the whole ontology, will be needed if we use tree method
  LabelSamples = LabelDescLoader.get_examples(args.data_dir, label_array=label_in_ontology)
  LabelSamples = GCN_data_loader.convert_labels_to_features(LabelSamples, MAX_SEQ_LEN_LABEL_DEF, Vocab, all_name_array=label_in_ontology, tokenize_style='space')

else:
  ## only get vectors for labels we want.
  LabelSamples = LabelDescLoader.get_examples(args.data_dir, label_array=label_to_test)
  LabelSamples = GCN_data_loader.convert_labels_to_features(LabelSamples, MAX_SEQ_LEN_LABEL_DEF, Vocab, all_name_array=label_to_test, tokenize_style='space')


GO_loader_for_precomp, GO_name_for_precomp = GCN_data_loader.make_label_loader (LabelSamples,args.batch_size_bert,fp16=False) ## if we fix encoder, then we don't have to worry about batch size, should be able to handle 32 or even 64


## **** load protein data


if args.ontology is None:
  add_name = ""
else:
  add_name = '-' + args.ontology

train_loader = protSeqLoader.ProtLoader (args.data_dir, 'train'+add_name+'.tsv', all_name_array, MAX_SEQ_LEN, 'random', args, args.do_kmer, label_to_test)

dev_loader = protSeqLoader.ProtLoader (args.data_dir, 'dev'+add_name+'.tsv', all_name_array, MAX_SEQ_LEN, 'sequential', args, args.do_kmer, label_to_test)


## **** make model ****

other_params = {'dropout': 0.2,
                'metric_option': args.metric_option,
                'labeldesc_loader': None,
                'edge_index': None,
                'AdjacencyMatrix': AdjacencyMatrix,
                'label_to_test_index': label_to_test_index,
                'label_in_ontology_index': label_in_ontology_index
                }

pretrained_weight = None
if args.w2v_emb is not None:
  pretrained_weight = pickle.load(open(args.w2v_emb,'rb'))
  pretrained_weight.shape[0]
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



## **** make bilstm model

biLstm = bi_lstm_model.bi_lstm_sent_encoder( other_params['word_vec_dim'], args.bilstm_dim )


# cosine model
# **** in using cosine model, we are not using the training sample A->B then B not-> A
cosine_loss = biLSTM_encoder_model.cosine_distance_loss(args.bilstm_dim,args.def_emb_dim, args)

# entailment model
ent_model = biLSTM_entailment_model.entailment_model (2,args.bilstm_dim,args.def_emb_dim,weight=torch.FloatTensor([1.5,.75])) 

metric_pass_to_joint_model = {'entailment':ent_model, 'cosine':cosine_loss}


#* **** create bilstm model with yes/no classification ****
## init joint model
GOEncoder = None
if args.precomputed_vector != None:
  GOEncoder = biLSTM_encoder_model.ReadGOVecFromFile(args.precomputed_vector)
else:
  print('need precomputed vector file')
  exit()

print ('see go encoder')
print (GOEncoder)


if args.go_enc_model_load is not None:
  print ('\n\nload back best model for GO encoder {}'.format(args.go_enc_model_load))
  GOEncoder.load_state_dict( torch.load( args.go_enc_model_load ), strict=False )

if args.fix_go_emb and (args.go_vec_dim > 0):
  GOEncoder.cuda()
  with torch.no_grad():
    go_emb = None
    if args.precomputed_vector != None:
      go_emb = GOEncoder.forward(GO_name_for_precomp) ## go_emb is num_go x dim, @label_name is only needed if @fout_name is used
    
    else:
      print('need precomputed_vector file')
      exit()

    other_params['go_emb'] = F.normalize( torch.FloatTensor(go_emb),dim=1 ).cuda()



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

if args.model_choice == 'DeepGOFlatSeqProtHwayNotUseGo': ## meant to be used to show that we need GO vectors. it has almost exactly the same structure as @DeepGOFlatSeqProtHwayGo
  prot2seq_model = ProtSeq2GOModel.DeepGOFlatSeqProtHwayNotUseGo (ProtEncoder, GOEncoder, args, **other_params)

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

