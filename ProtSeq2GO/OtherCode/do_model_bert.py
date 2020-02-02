

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
import GCN.encoder.data_loader as LabelDescDataLoaderClass

os.chdir("/local/datdb/GOmultitask") 
import ProtSeq2GO.ProtSeq2GOModel as ProtSeq2GOModel ####
import ProtSeq2GO.ProtSeqLoader as ProtSeqLoader
import ProtSeq2GO.arg_input as arg_input
args = arg_input.get_args()
print (args)


random.seed(args.seed) ####
np.random.seed(args.seed)
torch.manual_seed(args.seed)


MAX_SEQ_LEN = 2001

os.chdir(args.main_dir)

full_label_name_array = pd.read_csv("go_name_in_obo.csv", header=None)
full_label_name_array = list (full_label_name_array[0])
args.num_label = len(full_label_name_array)


## **** supposed we want to do testing on subset, to repeat deepgo baseline
label_to_test = None
label_to_test_index = None

if (args.num_label_to_test == 0) and (args.label_subset_file is None):
  args.num_label_to_test = args.num_label


elif args.label_subset_file is not None:

  ## we must properly extract the labels wanted
  ## if we do not want to update GO encoder, then we don't actually need this "if" statement. we can directly remove legacy terms from @go_name_in_obo.csv when creating @full_label_name_array

  print ('\n\nloading this set of go terms {}'.format(args.label_subset_file))

  ## we can do legacy GO terms if we don't care about the definitions

  label_to_test = pd.read_csv(args.label_subset_file, header=None)
  label_to_test = sorted ( list (label_to_test[0]) ) ## must sort, because we sort @full_label_name_array. we want the same ordering if we use @edge_index and run GCN to get emb. ORDERING DOES NOT MATTER IF WE DON'T USE GO DEFINITIONS

  print ('\n\ntotal label to test before filter legacy terms {}'.format(len(label_to_test)))

  label_to_test = [l for l in label_to_test if l in full_label_name_array] ## legacy terms will not be found in newest data

  args.num_label_to_test = len(label_to_test)
  print ('total label to test {}\n\n'.format(len(label_to_test)))

  ## get indexing of label to test from @full_label_name_array
  ## **** must be using the indexing from @full_label_name_array because GCN requires the graph, and @edge_index is made from this @full_label_name_array
  ## **** BERT will not need this graph, so for bert, maybe we just replace @full_label_name_array with @label_to_test ?

  # label_to_test_index = np.array ( [full_label_name_array.index(label) for label in label_to_test] )
  label_to_test_index = np.arange( args.num_label_to_test ) ## extract all labels


## **** create maxpool just like how deepgo does it. ??, then we need to load the entire BP/CC/MF
AdjacencyMatrix = None
label_in_ontology_index = None

if args.tree :

  label_in_ontology = pd.read_csv(args.label_in_ontology, header=None)
  label_in_ontology = sorted (list ( label_in_ontology[0] ) )

  # create label_in_ontology_index
  label_in_ontology_index = np.array ( [full_label_name_array.index(label) for label in label_in_ontology] ) ## used to extract terms in MF or BP or CC from a large @go_emb

  args.num_label_to_test = len(label_in_ontology)
  print ('because using tree, we need to add children terms, update total label to test {}\n\n'.format(len(label_in_ontology)))

  ## redo @label_to_test_index so that it matches the indexing of @label_in_ontology
  ## conditioned on @label_in_ontology, what is the index of each label to test.
  label_to_test_index = np.array ( [label_in_ontology.index(label) for label in label_to_test] )

  AdjacencyMatrix = pickle.load ( gzip.open(args.adjacency_matrix+".gzip.pickle","rb") )
  AdjacencyMatrix = torch.FloatTensor(AdjacencyMatrix)



## BERT requires labels to be read as "indexing"

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.modeling import BertForMaskedLM, BertConfig, BertForPreTraining
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule

## **** load label description
import BERT.encoder.arg_input as BERT_arg_input
import BERT.encoder.data_loader as BERT_data_loader
import BERT.encoder.encoder_model as BERT_encoder_model

MAX_SEQ_LEN_LABEL_DEF = 512 ## max len for GO def (probably can be smaller)

tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=True)

## reading in feature label is in @GCN folder. too lazy to port this function out.
LabelDescLoader = LabelDescDataLoaderClass.LabelProcessor()

if args.tree:
  # @label_in_ontology to get GO in the whole ontology, will be needed if we use tree method
  LabelSamples = LabelDescLoader.get_examples(args.data_dir, label_array=label_in_ontology)
  LabelSamples = LabelDescDataLoaderClass.LabelDescription2FeatureInput(LabelSamples, MAX_SEQ_LEN_LABEL_DEF, tokenizer, full_label_name_array=label_in_ontology, tokenize_style='bert')

else:
  ## only get BERT vectors for labels we want.
  LabelSamples = LabelDescLoader.get_examples(args.data_dir, label_array=label_to_test)
  LabelSamples = LabelDescDataLoaderClass.LabelDescription2FeatureInput(LabelSamples, MAX_SEQ_LEN_LABEL_DEF, tokenizer, full_label_name_array=label_to_test, tokenize_style='bert')


GO_loader_for_BERT, GO_name_for_BERT = LabelDescDataLoaderClass.LakeLabelDescLoader (LabelSamples,args.batch_size_pretrain_bert,fp16=False) ## if we fix BERT encoder, then we don't have to worry about batch size, should be able to handle 32 or even 64


## **** load protein data

# full_label_name_array = [ re.sub(r"GO:","",g) for g in full_label_name_array ] ## don't use GO: in the input files

if args.ontology is None:
  add_name = ""
else:
  add_name = '-' + args.ontology

train_loader = ProtSeqLoader.ProtLoader4TrainDev (args.data_dir, 'train'+add_name+'.tsv', full_label_name_array, MAX_SEQ_LEN, 'random', args, args.do_kmer, label_to_test)

dev_loader = ProtSeqLoader.ProtLoader4TrainDev (args.data_dir, 'dev'+add_name+'.tsv', full_label_name_array, MAX_SEQ_LEN, 'sequential', args, args.do_kmer, label_to_test)


## **** make model ****

# nonlinear_gcnn = {'tanh':F.tanh, 'relu':F.relu}

other_params = {'dropout': 0.2,
                'metric_option': args.metric_option,
                # 'nonlinear_gcnn': nonlinear_gcnn[args.nonlinear_gcnn],
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
  quant25, quant75 = ProtSeqLoader.GetNumObsPerQuantile(GO_counter)
  betweenQ25Q75 = ProtSeqLoader.IndexInRangeQuantileXY(label_to_test,GO_counter,quant25,quant75)
  quant25 = ProtSeqLoader.IndexBelowQuantileX(label_to_test,GO_counter,quant25)
  quant75 = ProtSeqLoader.IndexOverQuantileX(label_to_test,GO_counter,quant75)

  print ('counter 25 and 75 quantiles {} {}'.format(len(quant25), len(quant75)))

  other_params['GoCount'] = GO_counter
  other_params['quant25'] = quant25
  other_params['quant75'] = quant75
  other_params['betweenQ25Q75'] = betweenQ25Q75



## **** make BERT model

# bert language mask + next sentence model

bert_config = BertConfig( os.path.join(args.bert_model,"bert_config.json") )
cache_dir = args.cache_dir if args.cache_dir else os.path.join(
  str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed_{}'.format(args.local_rank))
bert_lm_sentence = BertForPreTraining.from_pretrained(args.bert_model,cache_dir=cache_dir)

# cosine model
# **** in using cosine model, we are not using the training sample A->B then B not-> A
cosine_loss = BERT_encoder_model.cosine_distance_loss(bert_config.hidden_size,args.def_emb_dim, args)
metric_pass_to_joint_model = {'entailment':None, 'cosine':cosine_loss}

#* **** add yes/no classifier to BERT ****
## init joint model
GOEncoder = BERT_encoder_model.encoder_model (bert_lm_sentence, metric_pass_to_joint_model[args.metric_option] , args, tokenizer, **other_params )

if args.go_enc_model_load is not None:
  print ('\n\nload back best model for GO encoder {}'.format(args.go_enc_model_load))
  GOEncoder.load_state_dict( torch.load( args.go_enc_model_load ), strict=False )

if args.fix_go_emb and (args.go_vec_dim > 0):
  GOEncoder.cuda()
  with torch.no_grad():
    print ('fix go vector emb, so we precompute to save time')
    go_emb = GOEncoder.write_label_vector(GO_loader_for_BERT,fout_name=None,label_name=None) ## go_emb is num_go x dim, @label_name is only needed if @fout_name is used
    print ('dim of go vectors {}'.format(go_emb.shape))
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

if args.model_choice == 'DeepGOFlatSeqProtHwayGoNotUsePPI':
  prot2seq_model = ProtSeq2GOModel.DeepGOFlatSeqProtHwayGoNotUsePPI (ProtEncoder, GOEncoder, args, **other_params)

if args.model_choice == 'DeepGOTreeSeqProtHwayGo':
  prot2seq_model = ProtSeq2GOModel.DeepGOTreeSeqProtHwayGo (ProtEncoder, GOEncoder, args, **other_params)

if args.model_choice == 'DeepGOFlatSeqProtConcatGo': ##!! concat go vector using pointwise style
  prot2seq_model = ProtSeq2GOModel.DeepGOFlatSeqProtConcatGo (ProtEncoder, GOEncoder, args, **other_params)

if args.model_choice == 'DeepGOFlatSeqConcatGo': ##!! concat go vector using pointwise style
  prot2seq_model = ProtSeq2GOModel.DeepGOFlatSeqConcatGo (ProtEncoder, GOEncoder, args, **other_params)



if args.prot2seq_model_load is not None:
  print ('\n\nload back best model to continue train {}'.format(args.prot2seq_model_load))
  prot2seq_model.load_state_dict( torch.load( args.prot2seq_model_load ), strict=False )


print ('\n\nsee model')
print (prot2seq_model)
prot2seq_model.cuda()


## **** train
if not args.not_train:
  prot2seq_model.do_train(train_loader, dev_loader, **other_params)


#### run on test set

print ('\n\nload back best model based on dev set {}'.format(os.path.join(args.result_folder,"best_state_dict.pytorch")))
prot2seq_model.load_state_dict( torch.load( os.path.join(args.result_folder,"best_state_dict.pytorch") ), strict=False )

## load test set
if args.test_data_name is None :
  test_loader = ProtSeqLoader.ProtLoader4TrainDev (args.data_dir, 'test'+add_name+'.tsv', full_label_name_array, MAX_SEQ_LEN, 'sequential', args, args.do_kmer, label_to_test)
else:
  test_loader = ProtSeqLoader.ProtLoader4AnyInputText (args.test_data_name, full_label_name_array, MAX_SEQ_LEN, 'sequential', args, args.do_kmer, label_to_test)

print ('\non test set\n')
result, preds, tr_loss = prot2seq_model.do_eval(test_loader, **other_params)
print ('dim of prediction {}'.format(preds['prediction'].shape))
if args.test_output_name is None :
  ## COMMENT save test set so that we can do analysis later
  pickle.dump( preds, open( os.path.join(args.result_folder, "prediction_testset.pickle"),"wb") )
else:
  pickle.dump( preds, open( os.path.join(args.result_folder, args.test_output_name+".pickle"),"wb") )

