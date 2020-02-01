
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

sys.path.append("/local/datdb/GOmultitask")

import biLSTM.encoder.arg_input as arg_input
args = arg_input.get_args()

import biLSTM.encoder.data_loader as data_loader
import biLSTM.encoder.encoder_model as encoder_model
import biLSTM.encoder.entailment_model as entailment_model
import biLSTM.encoder.bi_lstm_model as bi_lstm_model

MAX_SEQ_LEN = 256

os.chdir(args.main_dir)

full_label_name_array = pd.read_csv("go_name_in_obo.csv", header=None)
full_label_name_array = list (full_label_name_array[0])
args.num_label = len(full_label_name_array)

#### load label description data

if args.w2v_emb is not None: ## we can just treat each node as a vector without word description 
  Vocab = load_vocab(args.vocab_list) # all words found in pubmed and trained in w2v ... should trim down

## read go terms entailment pairs to train

InputReader = data_loader.QnliProcessor()
label_list = InputReader.get_labels() ## no/yes entailment style
num_labels = len(label_list) ## no/yes entailment style, not the total # node label

if args.test_file is None: 

  # full_label_name_array = [ re.sub(r"GO:","",g) for g in full_label_name_array ] ## **** for the rest, we don't use the "GO:"

  ## get label-label entailment data
  train_label_examples = InputReader.get_train_examples(args.qnli_dir,"train"+"_"+args.metric_option+".tsv")
  train_label_features = data_loader.StringInput2FeatureInput(train_label_examples, label_list, MAX_SEQ_LEN, tokenizer=Vocab, tokenize_style="space", full_label_name_array=full_label_name_array)
  train_label_dataloader = data_loader.MakeDataLoader4Model (train_label_features,batch_size=args.batch_size_aa_go,fp16=args.fp16, sampler='random',metric_option=args.metric_option)
  # torch.save( train_label_dataloader, os.path.join(args.qnli_dir,"train_label_dataloader"+name_add_on+".pytorch") )
  print ('\ntrain_label_examples {}'.format(len(train_label_examples))) # train_label_examples 35776

  """ get dev or test set  """
  # get label-label entailment data
  InputReader = data_loader.QnliProcessor()
  dev_label_examples = InputReader.get_dev_examples(args.qnli_dir,"dev"+"_"+args.metric_option+".tsv")
  dev_label_features = data_loader.StringInput2FeatureInput(dev_label_examples, label_list, MAX_SEQ_LEN, tokenizer=Vocab, tokenize_style="space", full_label_name_array=full_label_name_array)
  dev_label_dataloader = data_loader.MakeDataLoader4Model (dev_label_features,batch_size=args.batch_size_aa_go,fp16=args.fp16, sampler='sequential',metric_option=args.metric_option)
  # torch.save( dev_label_dataloader, os.path.join( args.qnli_dir, "dev_label_dataloader"+name_add_on+".pytorch") )
  print ('\ndev_label_examples {}'.format(len(dev_label_examples))) # dev_label_examples 7661


#### make model
other_params = {'dropout': 0.2,
                'metric_option': args.metric_option
                }

pretrained_weight = None
if args.w2v_emb is not None:
  pretrained_weight = pickle.load(open(args.w2v_emb,'rb'))
  pretrained_weight.shape[0]
  other_params ['num_of_word'] = pretrained_weight.shape[0]
  other_params ['word_vec_dim'] = pretrained_weight.shape[1]
  other_params ['pretrained_weight'] = pretrained_weight 

# cosine model
# **** in using cosine model, we are not using the training sample A->B then B not-> A
cosine_loss = encoder_model.cosine_distance_loss(args.bilstm_dim,args.def_emb_dim, args)

# entailment model
ent_model = entailment_model.entailment_model (num_labels,args.bilstm_dim,args.def_emb_dim,weight=torch.FloatTensor([1.5,.75])) # torch.FloatTensor([1.5,.75])

metric_pass_to_joint_model = {'entailment':ent_model, 'cosine':cosine_loss}

## make bilstm-entailment model

biLstm = bi_lstm_model.bi_lstm_sent_encoder( other_params['word_vec_dim'], args.bilstm_dim )

model = encoder_model.encoder_model ( args, metric_pass_to_joint_model[args.metric_option], biLstm, **other_params )

if args.use_cuda:
  print ('\n\n send model to gpu\n\n')
  model.cuda()


random.seed(args.seed) ####
np.random.seed(args.seed)
torch.manual_seed(args.seed)

print ('model is')
print (model)

if args.model_load is not None: 
  print ('\n\nload back best model {}'.format(args.model_load))
  model.load_state_dict( torch.load( args.model_load ), strict=False )

if args.epoch > 0 : ## here we do training 
  ## **** train
  tr_loss = model.do_train(train_label_dataloader,dev_label_dataloader)
  # save
  torch.save(model.state_dict(), os.path.join(args.result_folder,"last_state_dict.pytorch"))
  ## load back best
  print ('\n\nload back best state dict\n\n')
  model.load_state_dict( torch.load( os.path.join(args.result_folder,"best_state_dict.pytorch") ) )



#### added code to write out the vector as .txt 

if args.write_vector: 
  print ('\n\nwrite emb')
  print ('\n\nwrite GO vectors into text, using format of python gensim library')
  AllLabelDesc = data_loader.LabelReaderToProduceVecOutput ()
  examples = AllLabelDesc.get_examples( args.label_desc_dir ) ## file @label_desc_dir is tab delim 
  examples = data_loader.LabelDescription2FeatureInput ( examples , MAX_SEQ_LEN, tokenizer=Vocab, tokenize_style="space", full_label_name_array=full_label_name_array )
  AllLabelLoader, GO_names = data_loader.LabelLoaderToProduceVecOutput(examples,64) ## should be able to handle 64 labels at once 
  label_emb = model.write_label_vector( AllLabelLoader,os.path.join(args.result_folder,"label_vector.txt"), GO_names )


#### load test data
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


