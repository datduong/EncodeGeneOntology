
from __future__ import absolute_import, division, print_function

import argparse,csv,logging,os,random,sys, pickle, gzip
import numpy as np
import json
import pandas as pd

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.init import xavier_uniform_

from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from torch.nn import CrossEntropyLoss, MSELoss
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.modeling import BertForMaskedLM, BertConfig, BertForPreTraining
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule

# sys.path.append("C:/Users/dat/Dropbox/GOmultitask")
sys.path.append("/local/datdb/GOmultitask")
# sys.path.append("/u/scratch/d/datduong/GOmultitask/")

import BERT.encoder.arg_input as arg_input
args = arg_input.get_args()

import BERT.encoder.data_loader as data_loader
import BERT.encoder.entailment_model as entailment_model
import BERT.encoder.encoder_model as encoder_model

random.seed(args.seed) ####
np.random.seed(args.seed)
torch.manual_seed(args.seed)

MAX_SEQ_LEN = 512

## MUST READ IN WHAT ARE THE GO TERMS TO BE USED.
os.chdir(args.main_dir)
full_label_name_array = pd.read_csv("go_name_in_obo.csv", header=None)
full_label_name_array = list (full_label_name_array[0])
print ('loading in go terms found in go.obo, terms count {}'.format(len(full_label_name_array)))

# use BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=True) # args.do_lower_case args.bert_tokenizer

name_add_on = ""
if args.fp16:
  name_add_on = "_fp16"

name_add_on = name_add_on + "_" + args.metric_option

#### get data GO1-->GO2 relationship type

InputReader = data_loader.QnliProcessor()
label_list = InputReader.get_labels()
num_labels = len(label_list)

if args.test_file is None: ## so we do training because test file not given

  train_label_examples = InputReader.get_train_examples(args.qnli_dir,"train"+"_"+args.metric_option+".tsv")
  train_label_features = data_loader.StringInput2FeatureInput(train_label_examples, label_list, MAX_SEQ_LEN, tokenizer, "classification",full_label_name_array)
  train_label_dataloader = data_loader.MakeDataLoader4Model (train_label_features,batch_size=args.batch_size_aa_go,fp16=args.fp16, sampler='random',metric_option=args.metric_option)
  print ('\ntrain_label_examples {}'.format(len(train_label_examples))) # train_label_examples 35776

  #### get dev or test set

  # get label-label entailment data
  InputReader = data_loader.QnliProcessor()
  dev_label_examples = InputReader.get_dev_examples(args.qnli_dir,"dev"+"_"+args.metric_option+".tsv")
  dev_label_features = data_loader.StringInput2FeatureInput(dev_label_examples, label_list, MAX_SEQ_LEN, tokenizer, "classification",full_label_name_array)
  dev_label_dataloader = data_loader.MakeDataLoader4Model (dev_label_features,batch_size=args.batch_size_aa_go-2,fp16=args.fp16, sampler='sequential',metric_option=args.metric_option)
  print ('\ndev_label_examples {}'.format(len(dev_label_examples))) # dev_label_examples 7661


#### make model

##!! bert model as implemented in original Bert paper.

bert_config = BertConfig( os.path.join(args.bert_model,"bert_config.json") )

cache_dir = args.cache_dir if args.cache_dir else os.path.join(
  str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed_{}'.format(args.local_rank))

bert_lm_sentence = BertForPreTraining.from_pretrained(args.bert_model,cache_dir=cache_dir)  # @num_labels is yes/no
if args.fp16:
  bert_lm_sentence.half() ## don't send to cuda, we will send to cuda with the joint model

##!! entailment model to measure 2 GO terms ... COMMENT: NOT YET SUPPORTED
ent_model = entailment_model.entailment_model (num_labels,bert_config.hidden_size,args.def_emb_dim,weight=torch.FloatTensor([1.5,.75])) # torch.FloatTensor([1.5,.75])

##!! cosine model to measure 2 GO terms
# **** in using cosine model, we are not using the training sample A->B then B not-> A
other = {'metric_option':args.metric_option}
cosine_loss = encoder_model.cosine_distance_loss(bert_config.hidden_size,args.def_emb_dim, args)

metric_pass_to_joint_model = {'entailment':ent_model, 'cosine':cosine_loss}


#### Put the original Bert implementation with the cosine/entailment metrics

##!! make accessories variables for bert LM mask

samples_per_epoch = []
for i in range(int(args.num_train_epochs_bert)): ## how many bert epoch
  epoch_file = args.pregenerated_data / f"epoch_{i}.json"
  metrics_file = args.pregenerated_data / f"epoch_{i}_metrics.json"
  if epoch_file.is_file() and metrics_file.is_file():
    metrics = json.loads(metrics_file.read_text())
    samples_per_epoch.append(metrics['num_training_examples'])
  else:
    if i == 0:
      exit("No training data was found!")
    print(f"Warning! There are fewer epochs of pregenerated data ({i}) than training epochs ({args.num_train_epochs_bert}).")
    print("This script will loop over the available data, but training diversity may be negatively impacted.")
    num_data_epochs = i
    break
else:
  num_data_epochs = int(args.num_train_epochs_bert)

if args.local_rank == -1 or args.no_cuda:
  device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
  n_gpu = torch.cuda.device_count()
else:
  torch.cuda.set_device(args.local_rank)
  device = torch.device("cuda", args.local_rank)
  n_gpu = 1
  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
  torch.distributed.init_process_group(backend='nccl')

logging.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(device, n_gpu, bool(args.local_rank != -1), args.fp16))

args.batch_size_pretrain_bert = args.batch_size_pretrain_bert // args.gradient_accumulation_steps

if args.bert_output_dir.is_dir() and list(args.bert_output_dir.iterdir()):
  logging.warning(f"Output directory ({args.bert_output_dir}) already exists and is not empty!")
args.bert_output_dir.mkdir(parents=True, exist_ok=True)

total_train_examples = 0
for i in range(int(args.num_train_epochs_bert)):
  # The modulo takes into account the fact that we may loop over limited epochs of data
  total_train_examples += samples_per_epoch[i % len(samples_per_epoch)]

num_train_optim_steps_bert = int(total_train_examples / args.batch_size_pretrain_bert / args.gradient_accumulation_steps)
if args.local_rank != -1:
  num_train_optim_steps_bert = num_train_optim_steps_bert // torch.distributed.get_world_size()


#### init optim step ... need to know how many batches

num_line = os.popen( 'wc -l ' + os.path.join(args.qnli_dir,"train"+"_"+args.metric_option+".tsv") ).readlines()[0].strip().split()[0]
num_observation_in_train = int(num_line)
print ("\n\nnum_observation_in_train{}".format(num_observation_in_train))

num_train_optim_steps_entailment = int( np.ceil ( np.ceil ( num_observation_in_train / args.batch_size_aa_go ) / args.gradient_accumulation_steps) ) * args.num_train_epochs_entailment + args.batch_size_aa_go


#### init params of model to be trained ... we will just use cosine metric

BertMaskLMCosineSimJointModel = encoder_model.encoder_model (bert_lm_sentence, metric_pass_to_joint_model[args.metric_option] , args, tokenizer, **other )

if args.fp16:
  if args.metric_option == 'cosine':
    print ('\n\ncosine has problem with fp16, do not use fp16 for cosine distance\n\n')
    exit()
  BertMaskLMCosineSimJointModel.half()

if args.use_cuda:
  BertMaskLMCosineSimJointModel.cuda()

if args.model_load is not None :
  BertMaskLMCosineSimJointModel.load_state_dict(torch.load(args.model_load))

print (BertMaskLMCosineSimJointModel)

#### train model

## COMMENT: we can do joint training of Phase1 and Phase2, but let's just do Phase1 and then do Phase2, otherwise the model runs very slow.
## also... we did not observe significant improvement when jointly train Phase1 and Phase2

print ("\n")
print (args)

if args.average_layer:
  print ('\n\nuse_2nd_last_layer, mean, then trained using cosine distance\n\n')

for repeat in range( args.epoch ):

  print('\n\nrepeat step {}'.format(repeat))

  if args.update_bert and (repeat != args.epoch-1):## don't do this for last epoch
    if repeat==1: ## do 0 [1] 2 (so should run 3 round)
      BertMaskLMCosineSimJointModel.update_bert(num_data_epochs, num_train_optim_steps_bert)  # here we run LM mask model
      num_train_epochs_bert = 1 ## do only once on every other run ?
      torch.cuda.empty_cache()
      # save
      # torch.save(BertMaskLMCosineSimJointModel.state_dict(), os.path.join(args.result_folder,"best_state_dict"+name_add_on+".pytorch"))

  if repeat == (args.epoch-1): ## run twice longer
    BertMaskLMCosineSimJointModel.args.num_train_epochs_entailment = args.num_train_epochs_entailment*2
    ##!! need to redefine the below
    num_train_optim_steps_entailment = int( np.ceil ( np.ceil ( num_observation_in_train / args.batch_size_aa_go ) / args.gradient_accumulation_steps) ) * (args.num_train_epochs_entailment*2) + args.batch_size_aa_go

  tr_loss = BertMaskLMCosineSimJointModel.train_label(train_label_dataloader,
                                          num_train_optim_steps_entailment,
                                          dev_dataloader=dev_label_dataloader)

  torch.cuda.empty_cache()
  # save
  torch.save(BertMaskLMCosineSimJointModel.state_dict(), os.path.join(args.result_folder,"last_state_dict"+name_add_on+".pytorch"))



#### write out GO vectors as .txt

if args.write_vector:
  print ('\n\nwrite GO vectors into text, using format of python gensim library')
  AllLabelDesc = data_loader.LabelReaderToProduceVecOutput ()
  examples = AllLabelDesc.get_examples( args.label_desc_dir ) ## file @label_desc_dir is tab delim
  examples = data_loader.LabelDescription2FeatureInput ( examples , MAX_SEQ_LEN, tokenizer )
  AllLabelLoader, GO_names = data_loader.LabelLoaderToProduceVecOutput(examples,64) ## should be able to handle 64 labels at once
  label_emb = BertMaskLMCosineSimJointModel.write_label_vector( AllLabelLoader,os.path.join(args.result_folder,"label_vector.txt"), GO_names )

print ('\n\nload back best model')
BertMaskLMCosineSimJointModel.load_state_dict( torch.load( os.path.join(args.model_load) ) )

##!! re-use code and variable names... we only swap out input name into "testset"

train_label_dataloader = None ## clear some space
train_label_examples = None

print ('\n\nload test set')
InputReader = data_loader.QnliProcessor()

if args.test_file is None:
  args.test_file = args.qnli_dir,"test"+"_"+args.metric_option+".tsv"

print ('\n\ntest file name{}'.format(args.test_file))

dev_label_examples = InputReader.get_test_examples(args.test_file)
dev_label_features = data_loader.StringInput2FeatureInput(dev_label_examples, label_list, MAX_SEQ_LEN, tokenizer, "classification",full_label_name_array)
dev_label_dataloader = data_loader.MakeDataLoader4Model (dev_label_features,batch_size=args.batch_size_aa_go,fp16=args.fp16, sampler='sequential',metric_option=args.metric_option)
torch.save( dev_label_dataloader, os.path.join( args.qnli_dir, "test_label_dataloader"+name_add_on+".pytorch") )
print ('\ntest_label_examples {}'.format(len(dev_label_examples))) # dev_label_examples 7661

print ('\n\neval on test')
result, preds = BertMaskLMCosineSimJointModel.eval_label(dev_label_dataloader)

if args.write_score is not None:
  print ('\n\nscore file name {}'.format(args.write_score))
  fout = open(args.write_score,"w")
  fout.write( 'score\n'+'\n'.join(str(s) for s in preds) )
  fout.close()


