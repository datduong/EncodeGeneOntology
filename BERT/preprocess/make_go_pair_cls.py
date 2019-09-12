

from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string

from tqdm import tqdm

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

import pickle, gzip, os, sys, re
import random
from random import shuffle
import numpy as np
import pandas as pd

sys.path.append ("/u/flashscratch/d/datduong/GOmultitask")

import helper
import process_data.data_object.label_data_object

sys.path.append("/u/flashscratch/d/datduong/GOmultitask/process_data/go_term_object")
import go_term_object

## using same pairs to compare to 2018 paper 

def def2dict(name="go_def_in_obo.tsv"):
  GOdef = pd.read_csv(name,dtype=str,sep="\t")
  ## we remove the GO: notation because the train.csv doesn't use this notation. 
  return {re.sub("GO:","",g) : d['def'].values.tolist() for g, d in GOdef.groupby('name')}



def main (work_dir,data_dir): 

  go_def = def2dict(work_dir+"/go_def_in_obo.tsv") 

  if not os.path.exists (data_dir+"/go_bert_cls/"): os.mkdir (data_dir+"/go_bert_cls/")

  for set_type in ['test','dev','train']: 

    fout = open (data_dir+"/go_bert_cls/"+set_type+"_entailment.tsv","w")
    fout.write("index\tquestion\tsentence\tgo1\tgo2\tlabel")
      
    df = pd.read_csv(data_dir+"/"+set_type+"_go_id_b2b_correct.tsv",sep="\t",dtype=str)
    for index, row in df.iterrows():
      label = 'entailment'
      if row[2] == 'neutral': 
        label = 'not_entailment'
      #
      if row[0] not in go_def: continue 
      if row[1] not in go_def: continue   

      d1 = " ".join(s for s in go_def[row[0]])
      d1 = re.sub(r"\t"," ",d1)
      d1 = re.sub(r"\n"," ",d1)

      d2 = " ".join(s for s in go_def[row[1]])
      d2 = re.sub(r"\t"," ",d2)
      d2 = re.sub(r"\n"," ",d2)

      d1 = d1.lower()
      d2 = d2.lower()
      
      ## we keep the same A-->B, or A not B
      fout.write ( "\n" + str(index) + "\t" + d1 + "\t" + d2 + "\t" + row[0] + "\t" + row[1] + "\t" + label ) 
      if label == 'not_entailment': 
        ## add in B not A, given that A not B, so that we can learn the ordering of the vector concat
        fout.write ( "\n" + str(index) + "\t" + d2 + "\t" + d1 + "\t" + row[1] + "\t" + row[0] + "\t" + label ) 

    #
    fout.close()
  


if len(sys.argv)<1: ## run script 
	print("Usage: \n")
	sys.exit(1)
else:
	main ( sys.argv[1], sys.argv[2] ) 

