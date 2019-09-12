
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

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.modeling import BertForMaskedLM, BertConfig, BertForPreTraining
from pytorch_pretrained_bert.tokenization import BertTokenizer, load_vocab
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule

from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

sys.path.append("/local/datdb/GOmultitask")

import BERT.encoder.encoder_model as BERT_encoder_model

import GCN.encoder.arg_input as arg_input
args = arg_input.get_args()

import GCN.encoder.data_loader_add_encoder as data_loader
import GCN.encoder.encoder_model as encoder_model

MAX_SEQ_LEN = 256

tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=True) 

os.chdir(args.main_dir)

name_add_on = ""
if args.fp16:
  name_add_on = "_fp16"

## **** load edges ****

edge_index = pickle.load ( open("adjacency_matrix_coo_format.pickle","rb") )
edge_index = torch.tensor( edge_index, dtype=torch.long)
if args.use_cuda:
  edge_index = edge_index.cuda()

print ('num of edges {}'.format(edge_index.shape))

all_name_array = pd.read_csv("go_name_in_obo.csv", header=None)
all_name_array = list (all_name_array[0])

args.num_label = len(all_name_array)

## **** load label description data ****

label_desc_dataloader = None


## read go terms entailment pairs to train

processor = data_loader.QnliProcessor()
label_list = processor.get_labels() ## no/yes entailment style
num_labels = len(label_list) ## no/yes entailment style, not the total # node label

if args.test_file is None:

  ## get label-label entailment data
  train_label_examples = processor.get_train_examples(args.qnli_dir,"train"+"_"+args.metric_option+".tsv")
  train_label_features = data_loader.convert_examples_to_features(train_label_examples, label_list, 512, tokenizer,all_name_array)
  train_label_dataloader = data_loader.make_data_loader (train_label_features,batch_size=args.batch_size_label,fp16=args.fp16, sampler='random',metric_option=args.metric_option)
  # torch.save( train_label_dataloader, os.path.join(args.qnli_dir,"train_label_dataloader"+name_add_on+".pytorch") )
  print ('\ntrain_label_examples {}'.format(len(train_label_examples))) # train_label_examples 35776

  """ get dev or test set  """
  # get label-label entailment data
  processor = data_loader.QnliProcessor()
  dev_label_examples = processor.get_dev_examples(args.qnli_dir,"dev"+"_"+args.metric_option+".tsv")
  dev_label_features = data_loader.convert_examples_to_features(dev_label_examples, label_list, 512, tokenizer,all_name_array)
  dev_label_dataloader = data_loader.make_data_loader (dev_label_features,batch_size=args.batch_size_label,fp16=args.fp16, sampler='sequential',metric_option=args.metric_option)
  # torch.save( dev_label_dataloader, os.path.join( args.qnli_dir, "dev_label_dataloader"+name_add_on+".pytorch") )
  print ('\ndev_label_examples {}'.format(len(dev_label_examples))) # dev_label_examples 7661


## **** make model ****

nonlinear_gcnn = {'tanh':F.tanh, 'relu':F.relu}

other_params = {'dropout': 0.2,
                'metric_option': args.metric_option,
                'nonlinear_gcnn': nonlinear_gcnn[args.nonlinear_gcnn]
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
      # why are there missing GO terms still ??? maybe these GO terms were not used in "axiom A is_a B" in Onto2vec ??
      # set 0 for GO terms not found .... ??? this is the only way. or we have to update the entire matrix
      if go in pretrained_weight:
        temp[index] = pretrained_weight[go]
    ## now we get word dim and so forth
    pretrained_weight = temp ## override

  ## keep these
  other_params ['num_of_word'] = pretrained_weight.shape[0]
  other_params ['word_vec_dim'] = pretrained_weight.shape[1]
  other_params ['pretrained_weight'] = pretrained_weight



# cosine model
# **** in using cosine model, we are not using the training sample A->B then B not-> A
cosine_loss = encoder_model.cosine_distance_loss(args.def_emb_dim,args.def_emb_dim, args) ## remember to turn on reduce flag ???

# entailment model
# ent_model = entailment_model.entailment_model (num_labels,args.gcnn_dim,args.def_emb_dim,weight=torch.FloatTensor([1.5,.75])) # torch.FloatTensor([1.5,.75])

metric_pass_to_joint_model = {'entailment':None, 'cosine':cosine_loss}


## NEED TO MAKE THE BERT MODEL

# use BERT tokenizer
bert_config = BertConfig( os.path.join(args.bert_model,"bert_config.json") )

other = {'metric_option':args.metric_option}
bert_lm_sentence = BertForPreTraining.from_pretrained(args.bert_model)  
bert_lm_ent_model = BERT_encoder_model.encoder_model (bert_lm_sentence, metric_pass_to_joint_model[args.metric_option] , args, tokenizer, **other )



## make GCN model

model = encoder_model.encoder_with_bert ( args, bert_lm_ent_model, metric_pass_to_joint_model[args.metric_option], **other_params )


print ('\nmodel is\n')
print (model)
if args.use_cuda:
  print ('\n\n send model to gpu\n\n')
  model.cuda()


##

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

if args.model_load is not None:
  print ('\n\nload back best model')
  model.load_state_dict( torch.load( args.model_load ), strict=False )

if args.epoch > 0 : ## here we do training

  ## compute now many train iteration 
  num_observation_in_train = len(train_label_examples)
  num_train_optimization_steps = int( np.ceil ( np.ceil ( num_observation_in_train / args.batch_size_label ) / args.gradient_accumulation_steps) ) * args.batch_size_label + args.batch_size_label

  print ('num_observation_in_train {}'.format(num_observation_in_train))
  print ('num_train_optimization_steps {}'.format(num_train_optimization_steps))

  ## **** train
  tr_loss = model.do_train(train_label_dataloader,label_desc_dataloader,edge_index,num_train_optimization_steps,dev_label_dataloader)
  # save
  torch.save(model.state_dict(), os.path.join(args.result_folder,"last_state_dict"+name_add_on+".pytorch"))
  ## load back best
  print ('\n\nload back best state dict\n\n')
  model.load_state_dict( torch.load( os.path.join(args.result_folder,"best_state_dict"+name_add_on+".pytorch") ) )



print ('\n\nload test data\n\n')

# get label-label entailment data
processor = data_loader.QnliProcessor()

if args.test_file is None:
  args.test_file = args.qnli_dir,"test"+"_"+args.metric_option+".tsv"
  dev_label_examples = processor.get_dev_examples(args.test_file)
else:
  dev_label_examples = processor.get_test_examples(args.test_file)

print ('\n\ntest file name{}'.format(args.test_file))

dev_label_features = data_loader.convert_examples_to_features(dev_label_examples, label_list, all_name_array)
dev_label_dataloader = data_loader.make_data_loader (dev_label_features,batch_size=args.batch_size_label,fp16=args.fp16, sampler='sequential',metric_option=args.metric_option)


print ('\ntest_label_examples {}'.format(len(dev_label_examples))) # dev_label_examples 7661

print ('\n\neval on test')
result, preds, loss = model.do_eval(dev_label_dataloader,label_desc_dataloader,edge_index)

if args.write_score is not None:
  print ('\n\nscore file name {}'.format(args.write_score))
  fout = open(args.write_score,"w")
  fout.write( 'score\n'+'\n'.join(str(s) for s in preds) )
  fout.close()


