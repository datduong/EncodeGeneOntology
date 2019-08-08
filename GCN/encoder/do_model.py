
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

sys.path.append("/local/lgai/EncodeGeneOntology")

import GCN.encoder.arg_input as arg_input
args = arg_input.get_args()

import GCN.encoder.data_loader as data_loader
import GCN.encoder.encoder_model as encoder_model
#import GCN.encoder.entailment_model as entailment_model


MAX_SEQ_LEN = 256

os.chdir(args.main_dir)

name_add_on = ""
if args.fp16:
  name_add_on = "_fp16"

## **** load edges ****

# Note: edge_index is compressed adjacency matrix 
# Order of GO terms matches that in all_name_array
# Any pretrained weights must also have same order 
edge_index = pickle.load ( open("adjacency_matrix_coo_format.pickle","rb") )
edge_index = torch.tensor( edge_index, dtype=torch.long)
if args.use_cuda:
  edge_index = edge_index.cuda()

print ('num of edges {}'.format(edge_index.shape))



all_name_array = pd.read_csv("go_name_in_obo.csv", header=None)
all_name_array = list (all_name_array[0])

args.num_label = len(all_name_array)


## **** load label description data ****

if args.w2v_emb is not None: ## we can just treat each node as a vector without word description 

  Vocab = load_vocab(args.vocab_list) # all words found in pubmed and trained in w2v ... should trim down
  processor = data_loader.LabelProcessor()
  label_desc_examples = processor.get_examples(args.main_dir, all_name_array) ## must get all labels 
  
  label_desc_features = data_loader.convert_labels_to_features(label_desc_examples,max_seq_length=MAX_SEQ_LEN, tokenizer=Vocab, all_name_array=all_name_array,tokenize_style="space")

  if args.batch_size_label_desc == 0: 
    args.batch_size_label_desc = args.num_label

  label_desc_dataloader, label_desc_name = data_loader.make_label_loader (label_desc_features,batch_size=args.batch_size_label_desc,fp16=args.fp16) # len(label_desc_examples)

  print ('num of label desc to be transformed by gcn {}'.format ( len(label_desc_examples) )  )

else: 
  label_desc_dataloader = None

## **** for the rest, we don't use the "GO:"
# all_name_array = [ re.sub(r"GO:","",g) for g in all_name_array ]


## read go terms entailment pairs to train

processor = data_loader.QnliProcessor()
label_list = processor.get_labels() ## no/yes entailment style
num_labels = len(label_list) ## no/yes entailment style, not the total # node label

if args.test_file is None: 

  # all_name_array = [ re.sub(r"GO:","",g) for g in all_name_array ]

  ## get label-label entailment data
  train_label_examples = processor.get_train_examples(args.qnli_dir,"train"+"_"+args.metric_option+".tsv")
  train_label_features = data_loader.convert_examples_to_features(train_label_examples, label_list, all_name_array)
  train_label_dataloader = data_loader.make_data_loader (train_label_features,batch_size=args.batch_size_label,fp16=args.fp16, sampler='random',metric_option=args.metric_option)
  # torch.save( train_label_dataloader, os.path.join(args.qnli_dir,"train_label_dataloader"+name_add_on+".pytorch") )
  print ('\ntrain_label_examples {}'.format(len(train_label_examples))) # train_label_examples 35776

  """ get dev or test set  """
  # get label-label entailment data
  processor = data_loader.QnliProcessor()
  dev_label_examples = processor.get_dev_examples(args.qnli_dir,"dev"+"_"+args.metric_option+".tsv")
  dev_label_features = data_loader.convert_examples_to_features(dev_label_examples, label_list, all_name_array)
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
  if args.w2v_emb[-4:] == '.txt':
    # assume first line is a header
    with open(args.w2v_emb) as f:
      num_of_word, word_vec_dim = [int(x) for x in f.readline().rstrip().split()]

    pre_def_emb_dim = word_vec_dim # excludes any unpretrained dim

    pretrained_weight_names_txt = [] # GO terms in order in .txt
    #pretrained_weight_txt = np.zeros((num_of_word, word_vec_dim))
    pretrained_weight_d_txt = {} # dict[name] = weight arr
    with open(args.w2v_emb) as f:
      f.readline(); i = 0
      for line in f:
        lineL = line.rstrip().split()
        pretrained_weight_names_txt.append(lineL[0])
        weight = [float(x) for x in lineL[1:]]
        #pretrained_weight_txt[i,:] = weight
        pretrained_weight_d_txt[lineL[0]] = weight
        i += 1
    
    # check order of GO terms matches
    # if not, reorder and put 0's as placeholders for labels without a pretrained vec (TODO not sure how else to handle missing)
    if pretrained_weight_names_txt != list(all_name_array):
      print('num pretrained %d, num all name %d' % (len(pretrained_weight_names_txt), len(all_name_array)) ) 
      pretrained_weight = np.zeros( (args.num_label, pre_def_emb_dim) )
      missing_go_msg = "The following GO terms were missing and had 0s put in as a place holder: [ "
      for i in range(args.num_label):
        name = all_name_array[i]
        if name in pretrained_weight_d_txt:
          pretrained_weight[i,:] = pretrained_weight_d_txt[name]
        else:
          missing_go_msg += " %s " % name
          pretrained_weight[i,:] = np.zeros((pre_def_emb_dim,))
      missing_go_msg += " ]\n"
      print(missing_go_msg)

  else:
    pretrained_weight = pickle.load(open(args.w2v_emb,'rb')) 
    pretrained_weight.shape[0]
  
  other_params ['num_of_word'] = pretrained_weight.shape[0]
  other_params ['word_vec_dim'] = pretrained_weight.shape[1]
  other_params ['pretrained_weight'] = pretrained_weight 

# cosine model
# **** in using cosine model, we are not using the training sample A->B then B not-> A
cosine_loss = encoder_model.cosine_distance_loss(args.gcnn_dim,args.gcnn_dim, args)

# entailment model
# ent_model = entailment_model.entailment_model (num_labels,args.gcnn_dim,args.def_emb_dim,weight=torch.FloatTensor([1.5,.75])) # torch.FloatTensor([1.5,.75])

# comment out entailment model
#metric_pass_to_joint_model = {'entailment':ent_model, 'cosine':cosine_loss}
metric_pass_to_joint_model = {'cosine':cosine_loss} # TODO entailment commented out

## make GCN model

if args.w2v_emb is None:
  model = encoder_model.encoder_model ( args, metric_pass_to_joint_model[args.metric_option], **other_params )

else: 

  print (other_params)
  print ("\nusing these mode to model gcn input {}".format(args.word_mode))
  print ('word mode: %s\n' % args.word_mode)
  if args.word_mode == 'bilstm': 
    import GCN.encoder.bi_lstm_model as bi_lstm_model
    biLstm = bi_lstm_model.bi_lstm_sent_encoder( other_params['word_vec_dim'], args.bilstm_dim )
    if args.use_cuda: 
      biLstm.cuda() 
      
    model = encoder_model.encoder_model_biLstm ( args, metric_pass_to_joint_model[args.metric_option], biLstm, **other_params )

  if args.word_mode == 'conv1d': 
    model = encoder_model.encoder_model_conv1d ( args, metric_pass_to_joint_model[args.metric_option], **other_params )

  if args.word_mode == 'avepool': 
    model = encoder_model.encoder_model_avepool ( args, metric_pass_to_joint_model[args.metric_option], **other_params )

  if args.word_mode == 'pretrained':
    model = encoder_model.encoder_model_pretrained_embedding ( args, metric_pass_to_joint_model[args.metric_option], **other_params )

  if args.word_mode == 'pretrained_aux':
    model = encoder_model.encoder_model_extended_embedding ( args, metric_pass_to_joint_model[args.metric_option], **other_params )

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
  ## **** train
  tr_loss = model.do_train(train_label_dataloader,label_desc_dataloader,edge_index,dev_label_dataloader)
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
  dev_label_examples = processor.get_dev_examples(args.test_file[0],args.test_file[1])
else: 
  dev_label_examples = processor.get_test_examples(args.test_file[0],args.test_file[1])

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


