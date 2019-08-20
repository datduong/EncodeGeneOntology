
from __future__ import absolute_import, division, print_function

import argparse, csv, logging, os, random, sys, pickle, gzip, re
import numpy as np

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from pytorch_pretrained_bert.tokenization import BertTokenizer, load_vocab, whitespace_tokenize

sys.path.append('/u/flashscratch/d/datduong/protein-sequence-embedding-iclr2019/')
from src.alphabets import Uniprot21

logger = logging.getLogger(__name__)

## load protein sequences.

class InputFeatures(object):
  def __init__(self, label_id, input_id, input_mask, input_len, input_name, input_emb=None):
    self.label_id = label_id
    self.input_id = input_id
    self.input_mask = input_mask
    self.input_len = input_len
    self.input_name = input_name
    self.input_emb = input_emb ## protein emb from some other interaction network model


class InputExample(object):
  def __init__(self, guid, prot_name, label, aa_seq, prot_emb=None):
    self.guid = guid
    self.prot_name = prot_name
    self.label = label
    self.aa_seq = aa_seq
    self.prot_emb = prot_emb


class DataProcessor(object):
  """Base class for data converters for sequence classification data sets."""

  def get_train_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the train set."""
    raise NotImplementedError()

  def get_dev_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the dev set."""
    raise NotImplementedError()

  def get_labels(self):
    """Gets the list of labels for this data set."""
    raise NotImplementedError()

  @classmethod
  def _read_tsv(cls, input_file, quotechar=None):
    """Reads a tab separated value file."""
    with open(input_file, "r", encoding="utf-8") as f:
      reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
      lines = []
      for line in reader:
        lines.append(line)
    return lines

  @classmethod
  def _read_csv(cls, input_file, quotechar=None):
    """Reads a tab separated value file."""
    with open(input_file, "r", encoding="utf-8") as f:
      reader = csv.reader(f, delimiter=",", quotechar=quotechar)
      lines = []
      for line in reader:
        if sys.version_info[0] == 2:
          line = list(unicode(cell, 'utf-8') for cell in line)
        lines.append(line)
    return lines


class ProtProcessor(DataProcessor):

  def get_train_examples(self, data_dir, has_ppi_emb, name="train.tsv"):
    """See base class."""
    return self._create_examples(
      self._read_tsv(os.path.join(data_dir, name)), "train", has_ppi_emb)  # , "train.csv"

  def get_dev_examples(self, data_dir, has_ppi_emb, name="dev.tsv"):
    """See base class."""
    return self._create_examples(
      self._read_tsv(os.path.join(data_dir, name)), "dev_matched", has_ppi_emb)

  def get_test_examples(self, name, has_ppi_emb):
    """See base class."""
    return self._create_examples(self._read_tsv(name), "test_matched", has_ppi_emb)

  def _create_examples(self, lines, set_type, has_ppi_emb):
    # Entry Gene ontology IDs Sequence
    counter = 0
    examples = []

    for (i, line) in enumerate(lines):
      if i == 0:
        continue

      guid = "%s-%s" % (set_type, i)

      if has_ppi_emb :
        examples.append( InputExample(guid=guid, prot_name=line[0], label=line[1], aa_seq=line[2], prot_emb=line[3]) )

      else:
        examples.append( InputExample(guid=guid, prot_name=line[0], label=line[1], aa_seq=line[2] ) )

      counter = counter + 1

    return examples


MAX_SEQ_LEN_KMER = 1000
KMER_MAP={}
counter = 1 ## start at 1, so that 0 is padding
aa = 'ARNDCQEGHILKMFPSTWYV' ## do not include (B, O, J, U, X, Z) ... based on deepgo procedure
for a1 in aa:
  for a2 in aa:
    for a3 in aa:
      KMER_MAP[a1+a2+a3] = counter
      counter = counter + 1


def prot2kmer(aa_seq): ## 20^3 is 8000 kmer, add 1 for padding

  ## https://www.ddbj.nig.ac.jp/ddbj/code-e.html
  ## 'OUBZ' https://github.com/tbepler/protein-sequence-embedding-iclr2019/blob/master/src/alphabets.py#L56
  ## O=K
  ## B=N
  ## Z=Q
  ## U=C ? https://en.wikipedia.org/wiki/Selenocysteine
  ## X=unknown

  map_aa = {'O':'K', 'B':'N', 'Z':'Q', 'U':'C'}
  new_aa = ""
  for a in aa_seq:
    if a in map_aa:
      new_aa = new_aa + map_aa[a] ## map to simlar amino acid
    else:
      new_aa = new_aa + a

  kmer = []
  for i in range(len(new_aa)-3): ## RUNNING KMER BY STEP SIZE=1
    this_kmer = new_aa[ i : (i+3) ]
    if this_kmer in KMER_MAP:
      kmer.append ( KMER_MAP [ this_kmer ] ) ## append index of the k-mer
    else:
      kmer.append (0) ## unknown set as 0, so contribute nothing ?

  return kmer, len(kmer)


def prot2feature(examples, all_name_array, max_seq_length, do_kmer, subset_name_array, has_ppi_emb):

  alphabet = Uniprot21() ## convert string to indexing.
  # >>> alphabet.encode(b'ARNDCQEGHILKMFPSTWYVXOUBZ')
  # array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
  #       17, 18, 19, 20, 11,  4, 20, 20], dtype=uint8)

  # output is batch x max_len x dim when use @alphabet encoder
  # masking will be needed


  if subset_name_array is not None:
    print ('\nmodel runs only on subset of labels, so make 1-hot in subset of GO database, not all GO terms\n')
    all_name_array = subset_name_array

  label_map = {label : i for i, label in enumerate(all_name_array)}

  features = []

  for (ex_index, example) in tqdm(enumerate(examples)):

    if do_kmer:
      # print ('\nusing kmer with deepgo data will have seq max len 1000\n')
      max_seq_length = MAX_SEQ_LEN_KMER ## OVER RIDE
      input_id, input_len = prot2kmer (example.aa_seq)

    else:
      aa_seq = example.aa_seq[ 1:(len(example.aa_seq)-1) ]
      input_id = alphabet.encode(aa_seq.encode('utf-8'))
      input_id = list(input_id.astype(int))
      input_len = len(input_id)

    label = example.label.split(";") ## split 0016021;0031224;0044425
    label_id = np.zeros( len(label_map) )
    where_one = np.array ( [ label_map[g] for g in label if g in label_map ] )
    # if len(where_one) > 0:
    label_id [ where_one ] = 1 # 1-hot

    input_mask = [1] * input_len ## masking for batch mode

    ## !! be careful, because 0 in @alphabet is not a padding. we will need to use masking later
    padding = [0] * (max_seq_length - input_len)  # pad zero until max len
    input_id = input_id + padding
    input_mask = input_mask + padding

    if do_kmer:
      input_len = MAX_SEQ_LEN_KMER ## following deepgo, we take flatten, so we cannot have various lengths.

    if ex_index < 3:
      print ('\nsee sample {}'.format(ex_index))
      print ('sequence {}'.format(example.aa_seq))
      print ('input index {}'.format(input_id))
      print ('label index {}'.format(label_id))

    if has_ppi_emb:
      input_emb = [float(j) for j in example.prot_emb.split(";")] ## read this in a str, so convert to float. later conver to tensor
    else:
      input_emb = None

    features.append(InputFeatures(label_id=label_id,
                                  input_id=input_id, ## indexing of animo
                                  input_mask=input_mask,
                                  input_len=input_len,
                                  input_name=example.prot_name,
                                  input_emb=input_emb
                                  ) )

  return features


def makeProtLoader (train_features,batch_size,sampler,has_ppi_emb) :

  # must sort labels as they appear in @label_index_map

  all_name_ids = [f.input_name for f in train_features] # names as they appear in list
  all_input_ids = torch.tensor([f.input_id for f in train_features], dtype=torch.long)
  all_input_len = torch.tensor([f.input_len for f in train_features], dtype=torch.float)
  all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.float)
  all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.float)

  if has_ppi_emb:
    all_ppi_emb = torch.tensor([f.input_emb for f in train_features], dtype=torch.float)
  else:
    all_ppi_emb = torch.tensor([0 for f in train_features], dtype=torch.float)

  ## trim down
  all_input_ids = all_input_ids[ :, 0:int(max(all_input_len))]
  all_input_mask = all_input_mask[ :, 0:int(max(all_input_len))]

  train_data = TensorDataset(all_input_ids, all_input_len, all_input_mask, all_label_ids, all_ppi_emb)

  if sampler == 'random':
    train_sampler = RandomSampler(train_data)

  if sampler == 'sequential': ## meant for eval data
    train_sampler = SequentialSampler(train_data)

  train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

  return train_dataloader, all_name_ids


def ProtLoader (data_dir, data_type_name, all_name_array, max_seq_length, sampler, args, do_kmer, subset_name_array):

  processor = ProtProcessor()

  if 'train' in data_type_name:
    examples = processor.get_train_examples(data_dir,args.has_ppi_emb,data_type_name)

  else:
    examples = processor.get_dev_examples(data_dir,args.has_ppi_emb,data_type_name)

  features = prot2feature(examples, all_name_array, max_seq_length, do_kmer, subset_name_array,args.has_ppi_emb)
  dataloader, prot_name = makeProtLoader (features, batch_size=args.batch_size_label, sampler=sampler, has_ppi_emb=args.has_ppi_emb)
  return dataloader


### !!!!!
### !!!!!

# count go terms 
def IndexLessThanQuantile (label_to_test,count_dict,quant) : 
  index2extract = []
  for index, label in enumerate (label_to_test): 
    if label not in count_dict: ## label occur in whole data, but it doesn't occur in train data
      index2extract.append(index) ## see term GO:0004812
      continue
    if count_dict[label] < quant: 
      index2extract.append(index)
  return index2extract

def IndexMoreThanQuantile (label_to_test,count_dict,quant) : 
  index2extract = []
  for index, label in enumerate (label_to_test): 
    if label not in count_dict: ## label occur in whole data, but it doesn't occur in train data
      continue ## count is consider 0, so ... not greater than 75th quant
    if count_dict[label] > quant: 
      index2extract.append(index)
  return index2extract

def IndexBetweenQ25Q75Quantile (label_to_test,count_dict,quant1,quant2) : 
  index2extract = []
  for index, label in enumerate (label_to_test): 
    if label not in count_dict: ## label occur in whole data, but it doesn't occur in train data
      continue
    if (count_dict[label] >= quant1) and (count_dict[label] <= quant2): 
      index2extract.append(index)
  return index2extract

def GetCountQuantile (count_dict): 
  count = [count_dict[k] for k in count_dict] 
  quant = np.quantile(count,q=[0.25,0.75]) ## probably 2 of these are enough 
  return quant


