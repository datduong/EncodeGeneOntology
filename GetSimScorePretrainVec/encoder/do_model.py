
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

from torch_geometric.data import Data ## COMMENT we will need this if we load in pretrained GCN
from torch_geometric.nn import GCNConv

sys.path.append("/local/datdb/GOmultitask")

import GetSimScorePretrainVec.encoder.arg_input as arg_input
args = arg_input.get_args()
print (args)

import GetSimScorePretrainVec.encoder.data_loader as data_loader
import GetSimScorePretrainVec.encoder.encoder_model as encoder_model
import GetSimScorePretrainVec.encoder.entailment_model as entailment_model

MAX_SEQ_LEN = 256

os.chdir(args.main_dir)

full_label_name_array = pd.read_csv("go_name_in_obo.csv", header=None)
full_label_name_array = list (full_label_name_array[0])
args.num_label = len(full_label_name_array)

#### load label description data

Vocab = load_vocab(args.vocab_list) # all words found in pubmed and trained in w2v ... should trim down

#### read go terms, compute cosine score

InputReader = data_loader.QnliProcessor()
label_list = InputReader.get_labels() ## no/yes entailment style
num_labels = len(label_list) ## no/yes entailment style, not the total # node label

other_params = {'dropout': 0.2,
                'metric_option': args.metric_option
                }

if args.vector_file is None:
  print ('\n\nwe must have a pretrained GO vec.\n\n')
  exit() 


# cosine model
# **** in using cosine model, we are not using the training sample A->B then B not-> A
cosine_loss = encoder_model.cosine_distance_loss(args.bilstm_dim,args.def_emb_dim, args)

# entailment model
ent_model = entailment_model.entailment_model (num_labels,args.bilstm_dim,args.def_emb_dim,weight=torch.FloatTensor([1.5,.75])) # torch.FloatTensor([1.5,.75])

metric_pass_to_joint_model = {'entailment':ent_model, 'cosine':cosine_loss}

#### ##!! DO NOT NEED TO SPECIFY ANY GO ENCODER. we can do simple look-up from @pretrained_weight

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

model = encoder_model.encoder_model ( args, metric_pass_to_joint_model[args.metric_option], **other_params )
print ('model is')
print (model)
if args.use_cuda:
  print ('\n\n send model to gpu\n\n')
  model.cuda()

####

print ('\n\nload test data\n\n')

# get label-label entailment data
# WILL REUSE SOME VARIABLE NAMES

InputReader = data_loader.QnliProcessor()

if args.test_file is None:
  args.test_file = args.qnli_dir,"test"+"_"+args.metric_option+".tsv"
  dev_label_examples = InputReader.get_dev_examples(args.test_file)
else: 
  dev_label_examples = InputReader.get_test_examples(args.test_file)

print ('\n\ntest file name{}'.format(args.test_file))

dev_label_features = data_loader.StringInput2FeatureInput(dev_label_examples, label_list, MAX_SEQ_LEN, tokenizer=Vocab, tokenize_style="space", full_label_name_array=full_label_name_array)

dev_label_dataloader = data_loader.MakeDataLoader4Model (dev_label_features,batch_size=args.batch_size_aa_go,fp16=args.fp16, sampler='sequential',metric_option=args.metric_option)

print ('\ntest_label_examples {}'.format(len(dev_label_examples))) # dev_label_examples 7661

print ('\n\neval on test')
result, preds, loss = model.do_eval(dev_label_dataloader)

if args.write_score is not None: 
  print ('\n\nscore file name {}'.format(args.write_score))
  fout = open(args.write_score,"w")
  fout.write( 'score\n'+'\n'.join(str(s) for s in preds) )
  fout.close() 


