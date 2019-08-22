


## copy the BERT data loader


from __future__ import absolute_import, division, print_function

import argparse, csv, logging, os, random, sys, pickle, gzip, re
import numpy as np

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from pytorch_pretrained_bert.tokenization import BertTokenizer, load_vocab, whitespace_tokenize


logger = logging.getLogger(__name__)

# must match the token correctly. If use BERT tokenizer for labels, then the doctor notes must also use BERT.
# from nltk.tokenize import RegexpTokenizer
# tokenizer = RegexpTokenizer(r'\w+') # retain only alphanumeric


def bert_tokenizer_style(tokenizer,text_a, add_cls_sep=True):

  tokens_a = tokenizer.tokenize(text_a)

  ## first sentence
  # The convention in BERT is:
  # (a) For sequence pairs:
  #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
  #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
  # (b) For single sequences:
  #  tokens:   [CLS] the dog is hairy . [SEP]
  #  type_ids: 0   0   0   0  0     0 0
  #
  # Where "type_ids" are used to indicate whether this is the first
  # sequence or the second sequence. The embedding vectors for `type=0` and
  # `type=1` were learned during pre-training and are added to the wordpiece
  # embedding vector (and position vector). This is not *strictly* necessary
  # since the [SEP] token unambiguously separates the sequences, but it makes
  # it easier for the model to learn the concept of sequences.
  #
  # For classification tasks, the first vector (corresponding to [CLS]) is
  # used as as the "sentence vector". Note that this only makes sense because
  # the entire model is fine-tuned.

  if add_cls_sep:
    tokens_a = ["[CLS]"] + tokens_a + ["[SEP]"]

  input_1_ids = tokenizer.convert_tokens_to_ids(tokens_a) ## should we use the [CLS] to represent the sentence for downstream task ??

  return input_1_ids , len(input_1_ids)


def icd9_tokenizer_style (vocab,text_a):
  # we follow same tokenization approach in original paper
  # @Vocab is some object that convert words to index exactly in the order of the pretrained word vectors.
  # use Vocab = load_vocab('/local/datdb/MIMIC3database/format10Jan2019/vocab+icd_index_map.txt')
  # @text_a can be split by space (in case of preprocessed icd9 notes)
  tokens_a = whitespace_tokenize(text_a)
  input_ids = []
  for token in tokens_a:
    if token in vocab:
      input_ids.append(vocab[token])
    else:
      input_ids.append(1) # unknown
  return input_ids, len(input_ids)


class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self, input_1_ids, input_1_len, input_1_name, input_2_ids=None, input_2_len=None, input_2_name=None , input_ids=None, input_mask=None, segment_ids=None, label_id=None, input_1_mask=None, input_2_mask=None):
    self.input_1_ids = input_1_ids
    self.input_2_ids = input_2_ids
    self.input_1_len = input_1_len
    self.input_2_len = input_2_len
    self.input_1_name = input_1_name
    self.input_2_name = input_2_name
    self.label_id = label_id
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.input_1_mask = input_1_mask
    self.input_2_mask = input_2_mask

class InputExample(object):
  """A single training/test example for simple sequence classification."""

  def __init__(self, guid, text_a, name_a, text_b=None, name_b=None, label=None, text_len=None):
    """Constructs a InputExample.
    Args:
    guid: Unique id for the example.
    text_a: string. The untokenized text of the first sequence. For single
    sequence tasks, only this sequence must be specified.
    text_b: (Optional) string. The untokenized text of the second sequence.
    Only must be specified for sequence pair tasks.
    label: (Optional) string. The label of the example. This should be
    specified for train and dev examples, but not for test examples.
    """
    self.guid = guid
    self.text_a = text_a
    self.text_b = text_b
    self.name_a = name_a
    self.name_b = name_b
    self.label = label
    self.text_len = text_len

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
        if sys.version_info[0] == 2:
          line = list(unicode(cell, 'utf-8') for cell in line)
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

class QnliProcessor(DataProcessor):
  """Processor for the QNLI data set (GLUE version)."""

  def get_train_examples(self, data_dir, name="train.tsv"):
    """See base class."""
    return self._create_examples(
      self._read_tsv(os.path.join(data_dir, name)), "train")  # , "train.tsv"

  def get_dev_examples(self, data_dir, name="dev.tsv"):
    """See base class."""
    return self._create_examples(
      self._read_tsv(os.path.join(data_dir, name)), "dev_matched")

  def get_test_examples(self, test_file, name="dev.tsv"):
    """See base class."""
    return self._create_examples(
      self._read_tsv(test_file), "test_matched")

  def get_labels(self):
    """See base class."""
    return ["entailment", "not_entailment"] ## 0=entailment

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    counter = 0
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      guid = "%s-%s" % (set_type, line[0])
      text_a = line[1]
      text_b = line[2]
      name_a = line[3] # name are not used, but good for debug
      name_b = line[4]
      label = line[5]

      if 'GO' not in name_a:  ## STRICT ENFORCE GO:XYZ SYNTAX
        name_a = 'GO:'+name_a
      if 'GO' not in name_b: 
        name_b = 'GO:'+name_b

      examples.append(
        InputExample(guid=guid, text_a=text_a.lower(), text_b=text_b.lower(), name_a=name_a, name_b=name_b, label=label))
      counter = counter + 1
    return examples


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, tokenize_style="space", all_name_array=None):
  """Loads a data file into a list of `InputBatch`s."""

  label_map = {label : i for i, label in enumerate(label_list)}

  ## STRICT ENFORCE GO:XYZ SYNTAX

  # if all_name_array is not None: 
  #   all_name_array = [ re.sub("GO:","",g) for g in all_name_array ] ## the train.csv doesn't use the GO:xyz syntax

  features = []

  for (ex_index, example) in tqdm(enumerate(examples)):

    ## label
    try: ## some terms have changed, legacy input file
      input_pos=[all_name_array.index(example.name_a), all_name_array.index(example.name_b)]
    except:
      continue

    if tokenize_style == 'bert': 
      input_1_ids, input_1_len = bert_tokenizer_style(tokenizer, example.text_a, add_cls_sep=True)
    else: 
      input_1_ids, input_1_len = icd9_tokenizer_style(tokenizer, example.text_a)


    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_1_mask = [1] * len(input_1_ids)

    padding = [0] * (max_seq_length - len(input_1_ids))  # pad zero until max len
    input_1_ids = input_1_ids + padding
    input_1_mask = input_1_mask + padding

    assert len(input_1_ids) == max_seq_length
    assert len(input_1_mask) == max_seq_length

    if ex_index < 5:
      print("\n*** Label Description Example ***")
      print("guid: %s" % (example.guid))
      if tokenize_style == 'bert': 
        print("tokens: %s" % " ".join(
                [str(x) for x in tokenizer.tokenize(example.text_a)]))
      print("input_ids: %s" % " ".join([str(x) for x in input_1_ids]))
      print("input_mask: %s" % " ".join([str(x) for x in input_1_mask]))

    ## input 2
    if tokenize_style == 'bert': 
      input_2_ids, input_2_len = bert_tokenizer_style(tokenizer, example.text_b, add_cls_sep=True)
    else: 
      input_2_ids, input_2_len = icd9_tokenizer_style(tokenizer, example.text_b)


    input_2_mask = [1] * len(input_2_ids)

    padding = [0] * (max_seq_length - len(input_2_ids))  # pad zero until max len
    input_2_ids = input_2_ids + padding
    input_2_mask = input_2_mask + padding

    assert len(input_2_ids) == max_seq_length
    assert len(input_2_mask) == max_seq_length

    ## label
    label_id = label_map[example.label]

    features.append(InputFeatures(input_1_ids=input_1_ids,
                                  input_1_len=input_1_len,  # true len, not count in the 0-pad
                                  input_1_name=example.name_a,
                                  input_1_mask=input_1_mask,
                                  input_2_ids=input_2_ids,
                                  input_2_len=input_2_len,  # true len, not count in the 0-pad
                                  input_2_name=example.name_b,
                                  input_2_mask=input_2_mask,
                                  label_id=label_id
                                  ) )

  return features


def make_data_loader (train_features,batch_size,fp16=False,sampler='random',metric_option='cosine') :

  all_input_1_ids = torch.tensor([f.input_1_ids for f in train_features], dtype=torch.long)
  all_input_2_ids = torch.tensor([f.input_2_ids for f in train_features], dtype=torch.long)
  
  name_1 = [f.input_1_name for f in train_features]
  name_2 = [f.input_2_name for f in train_features]
  all_input_1_name = torch.tensor([int(f[3:]) for f in name_1], dtype=torch.int)
  all_input_2_name = torch.tensor([int(f[3:]) for f in name_2], dtype=torch.int)
  
  all_input_1_len = torch.tensor([f.input_1_len for f in train_features], dtype=torch.float)
  all_input_2_len = torch.tensor([f.input_2_len for f in train_features], dtype=torch.float)

  # add segment_ids and input_mask
  all_input_1_mask = torch.tensor([f.input_1_mask for f in train_features], dtype=torch.long)
  all_input_2_mask = torch.tensor([f.input_2_mask for f in train_features], dtype=torch.long)

  all_input_1_ids.data = all_input_1_ids.data[ : , 0:int(max(all_input_1_len)) ] # trim down input to max len of the batch
  all_input_1_mask.data = all_input_1_mask.data[ : , 0:int(max(all_input_1_len)) ] # trim down input to max len of the batch

  all_input_2_ids.data = all_input_2_ids.data[ : , 0:int(max(all_input_2_len)) ]
  all_input_2_mask.data = all_input_2_mask.data[ : , 0:int(max(all_input_2_len)) ]

  ## takes long for standard entailment model
  all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)

  if metric_option == 'cosine': ## revert label into the format -1=not_entailment and 1=entailment
    all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.float)
    all_label_ids.data[all_label_ids.data==1] = -1
    all_label_ids.data[all_label_ids.data==0] = 1 # @return ["entailment", "not_entailment"] ## 0=entailment
    if fp16:
      all_label_ids = all_label_ids.half()

  if fp16: ## don't need this if we configure BERT fp16 ? well it save some space
    all_input_1_len = all_input_1_len.half() ## 16 bit
    all_input_2_len = all_input_2_len.half()


  train_data = TensorDataset ( all_input_1_name, all_input_1_ids, all_input_1_len, all_input_1_mask, all_input_2_name, all_input_2_ids, all_input_2_len, all_input_2_mask, all_label_ids )

  if sampler == 'random':
    train_sampler = RandomSampler(train_data)

  if sampler == 'sequential': ## meant for eval data
    train_sampler = SequentialSampler(train_data)

  return DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)



class LabelProcessorForWrite(DataProcessor):
  """Processor for the QNLI data set (GLUE version)."""

  def get_examples(self, label_desc_dir):
    """See base class."""
    return self._create_examples(
      self._read_tsv(label_desc_dir), "train")  # , "train.tsv"

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    counter = 0
    examples = []
    for (i, line) in enumerate(lines):
      guid = "%s-%s" % (set_type, counter)
      text_a = line[1]
      name_a = line[0] # name are not used, but good for debug
      examples.append(
        InputExample(guid=guid, text_a=text_a.lower(), text_b=None, name_a=name_a, name_b=None, label=None))
      counter = counter + 1
    return examples


def convert_label_desc_to_features(examples, max_seq_length, tokenizer):
  """Loads a data file into a list of `InputBatch`s."""

  features = []

  for (ex_index, example) in tqdm(enumerate(examples)):

    input_1_ids, input_1_len = bert_tokenizer_style(tokenizer, example.text_a, add_cls_sep=True)
    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_1_mask = [1] * len(input_1_ids)

    padding = [0] * (max_seq_length - len(input_1_ids))  # pad zero until max len
    input_1_ids = input_1_ids + padding
    input_1_mask = input_1_mask + padding

    assert len(input_1_ids) == max_seq_length
    assert len(input_1_mask) == max_seq_length

    if ex_index < 5:
      print("\n*** Label Description Example ***")
      print("guid: %s" % (example.guid))
      print("tokens: %s" % " ".join([str(x) for x in tokenizer.tokenize(example.text_a)]))
      print("input_ids: %s" % " ".join([str(x) for x in input_1_ids]))
      print("input_mask: %s" % " ".join([str(x) for x in input_1_mask]))


    features.append(InputFeatures(input_1_ids=input_1_ids,
                                  input_1_len=input_1_len,  # true len, not count in the 0-pad
                                  input_1_name=example.name_a,
                                  input_1_mask=input_1_mask
                                  ) )

  return features


def label_loader_for_write (train_features,batch_size,fp16=False): 
  name_1 = [f.input_1_name for f in train_features]
  all_input_1_name = torch.tensor([int(f[3:]) for f in name_1], dtype=torch.int) ## same order as @train_sampler

  all_input_1_ids = torch.tensor([f.input_1_ids for f in train_features], dtype=torch.long)

  all_input_1_len = torch.tensor([f.input_1_len for f in train_features], dtype=torch.float)

  # add segment_ids and input_mask
  all_input_1_mask = torch.tensor([f.input_1_mask for f in train_features], dtype=torch.long)

  all_input_1_ids.data = all_input_1_ids.data[ : , 0:int(max(all_input_1_len)) ] # trim down input to max len of the batch
  all_input_1_mask.data = all_input_1_mask.data[ : , 0:int(max(all_input_1_len)) ] # trim down input to max len of the batch

  if fp16: ## don't need this if we configure BERT fp16 ? well it save some space
    all_input_1_len = all_input_1_len.half() ## 16 bit

  train_data = TensorDataset ( all_input_1_name, all_input_1_ids, all_input_1_len, all_input_1_mask )

  ## meant for eval data
  train_sampler = SequentialSampler(train_data)

  return DataLoader(train_data, sampler=train_sampler, batch_size=batch_size), all_input_1_name

