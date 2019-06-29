
from __future__ import absolute_import, division, print_function

import argparse,csv,logging,os,random,sys, pickle, gzip
import numpy as np
import json

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


MAX_SEQ_LEN = 512

# use BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=True) # args.do_lower_case args.bert_tokenizer

name_add_on = ""
if args.fp16:
  name_add_on = "_fp16"


name_add_on = name_add_on + "_" + args.metric_option

# get label-label entailment data
processor = data_loader.QnliProcessor()
label_list = processor.get_labels()
num_labels = len(label_list)
# try:
#   train_label_dataloader = torch.load( os.path.join(args.qnli_dir,"train_label_dataloader"+name_add_on+".pytorch") )
# except:

if args.test_file is None: 
  train_label_examples = processor.get_train_examples(args.qnli_dir,"train"+"_"+args.metric_option+".tsv")
  train_label_features = data_loader.convert_examples_to_features(train_label_examples, label_list, MAX_SEQ_LEN, tokenizer, "classification")
  train_label_dataloader = data_loader.make_data_loader (train_label_features,batch_size=args.batch_size_label,fp16=args.fp16, sampler='random',metric_option=args.metric_option)
  print ('\ntrain_label_examples {}'.format(len(train_label_examples))) # train_label_examples 35776


  """ get dev or test set  """

  # get label-label entailment data
  processor = data_loader.QnliProcessor()
  dev_label_examples = processor.get_dev_examples(args.qnli_dir,"dev"+"_"+args.metric_option+".tsv")
  dev_label_features = data_loader.convert_examples_to_features(dev_label_examples, label_list, MAX_SEQ_LEN, tokenizer, "classification")
  dev_label_dataloader = data_loader.make_data_loader (dev_label_features,batch_size=args.batch_size_label-2,fp16=args.fp16, sampler='sequential',metric_option=args.metric_option)
  # torch.save( dev_label_dataloader, os.path.join( args.qnli_dir, "dev_label_dataloader"+name_add_on+".pytorch") )
  print ('\ndev_label_examples {}'.format(len(dev_label_examples))) # dev_label_examples 7661


## **** make model ****

# bert model

bert_config = BertConfig( os.path.join(args.bert_model,"bert_config.json") )

cache_dir = args.cache_dir if args.cache_dir else os.path.join(
  str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed_{}'.format(args.local_rank))

bert_lm_sentence = BertForPreTraining.from_pretrained(args.bert_model,cache_dir=cache_dir)  # @num_labels is yes/no
if args.fp16:
  bert_lm_sentence.half() ## don't send to cuda, we will send to cuda with the joint model


# entailment model
ent_model = entailment_model.entailment_model (num_labels,bert_config.hidden_size,args.def_emb_dim,neighbor_dim=0,do_neighbor=False,weight=None) # torch.FloatTensor([1.5,.75])

# cosine model
# **** in using cosine model, we are not using the training sample A->B then B not-> A
other = {'metric_option':args.metric_option}
cosine_loss = encoder_model.cosine_distance_loss(bert_config.hidden_size,args.def_emb_dim, args)

metric_pass_to_joint_model = {'entailment':ent_model, 'cosine':cosine_loss}


# **** joint model ****

# make accessories variables for bert LM mask

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

args.batch_size_bert = args.batch_size_bert // args.gradient_accumulation_steps

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

if args.bert_output_dir.is_dir() and list(args.bert_output_dir.iterdir()):
  logging.warning(f"Output directory ({args.bert_output_dir}) already exists and is not empty!")
args.bert_output_dir.mkdir(parents=True, exist_ok=True)

total_train_examples = 0
for i in range(int(args.num_train_epochs_bert)):
  # The modulo takes into account the fact that we may loop over limited epochs of data
  total_train_examples += samples_per_epoch[i % len(samples_per_epoch)]

num_train_optim_steps_bert = int(total_train_examples / args.batch_size_bert / args.gradient_accumulation_steps)
if args.local_rank != -1:
  num_train_optim_steps_bert = num_train_optim_steps_bert // torch.distributed.get_world_size()


## init entailment model optim step

num_line = os.popen( 'wc -l ' + os.path.join(args.qnli_dir,"train"+"_"+args.metric_option+".tsv") ).readlines()[0].strip().split()[0]
num_observation_in_train = int(num_line)
print ("\n\nnum_observation_in_train{}".format(num_observation_in_train))

num_train_optim_steps_entailment = int( np.ceil ( np.ceil ( num_observation_in_train / args.batch_size_label ) / args.gradient_accumulation_steps) ) * args.num_train_epochs_entailment + args.batch_size_label

## init joint model
bert_lm_ent_model = encoder_model.encoder_model (bert_lm_sentence, metric_pass_to_joint_model[args.metric_option] , args, tokenizer, **other )

if args.fp16:
  if args.metric_option == 'cosine': 
    print ('\n\ncosine has problem with fp16, do not use fp16 for cosine distance\n\n')
    exit()
  bert_lm_ent_model.half()

if args.use_cuda:
  bert_lm_ent_model.cuda()

if args.model_load is not None :
  bert_lm_ent_model.load_state_dict(torch.load(args.model_load))

# if args.do_test and (not args.do_continue):

#   print ("\n\neval only, not train\n\n")

#   bert_lm_ent_model.eval()

#   ## eval on train data (should see very high accuracy)
#   print ("\neval on train data")
#   torch.cuda.empty_cache()
#   bert_lm_ent_model.eval_label(train_label_dataloader)

#   print ("\neval on dev data") ### 
#   torch.cuda.empty_cache()
#   bert_lm_ent_model.eval_label(dev_label_dataloader)

#   exit()

# if args.write_vector: 

#   processor = data_loader.LabelProcessorForWrite()
#   label_desc_examples = processor.get_examples(args.label_desc_dir)
#   label_desc_features = data_loader.convert_label_desc_to_features(label_desc_examples,max_seq_length=128, tokenizer=tokenizer)
#   label_desc_dataloader, label_desc_name = data_loader.label_loader_for_write (label_desc_features,batch_size=64,fp16=args.fp16)

#   fout_name = os.path.join(args.qnli_dir,"label_vector_"+args.metric_option+".tsv")
#   bert_lm_ent_model.write_label_vector (label_desc_dataloader,fout_name,label_desc_name)

#   exit() 


## **** train

for repeat in range( args.epoch ):

  print('\n\nrepeat step {}'.format(repeat))

  if args.update_bert: 
    if (repeat % 2) == 0:
      bert_lm_ent_model.update_bert(num_data_epochs, num_train_optim_steps_bert)  # here we run LM mask model
      num_train_epochs_bert = 1

      torch.cuda.empty_cache()
      # save
      torch.save(bert_lm_ent_model.state_dict(), os.path.join(args.result_folder,"best_state_dict"+name_add_on+".pytorch"))

  tr_loss = bert_lm_ent_model.train_label(train_label_dataloader,
                                                          num_train_optim_steps_entailment,
                                                          dev_dataloader=dev_label_dataloader)

  torch.cuda.empty_cache()

  # save
  torch.save(bert_lm_ent_model.state_dict(), os.path.join(args.result_folder,"last_state_dict"+name_add_on+".pytorch"))



print ('\n\nload back best model')
bert_lm_ent_model.load_state_dict( torch.load( os.path.join(args.result_folder,"best_state_dict.pytorch") ) )

""" get dev or test set  """

train_label_dataloader = None ## clear some space
train_label_examples = None 

print ('\n\nload test set')
processor = data_loader.QnliProcessor()

if args.test_file is None:
  args.test_file = args.qnli_dir,"test"+"_"+args.metric_option+".tsv"

print ('\n\ntest file name{}'.format(args.test_file))

dev_label_examples = processor.get_test_examples(args.test_file)
dev_label_features = data_loader.convert_examples_to_features(dev_label_examples, label_list, MAX_SEQ_LEN, tokenizer, "classification")
dev_label_dataloader = data_loader.make_data_loader (dev_label_features,batch_size=args.batch_size_label,fp16=args.fp16, sampler='sequential',metric_option=args.metric_option)
torch.save( dev_label_dataloader, os.path.join( args.qnli_dir, "test_label_dataloader"+name_add_on+".pytorch") )
print ('\ntest_label_examples {}'.format(len(dev_label_examples))) # dev_label_examples 7661


print ('\n\neval on test')
result, preds = bert_lm_ent_model.eval_label(dev_label_dataloader)

if args.write_score is not None: 
  print ('\n\nscore file name {}'.format(args.write_score))
  fout = open(args.write_score,"w")
  fout.write( 'score\n'+'\n'.join(str(s) for s in preds) )
  fout.close() 

