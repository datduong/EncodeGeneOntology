
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

## make data for BERT fine tune.
## take one branch as a "document"

work_dir = '/u/flashscratch/d/datduong/goAndGeneAnnotationDec2018/'
os.chdir(work_dir)

where_go_info ='w2vDim300/EntDataJan19w300Base'
go_def = pickle.load(open ( 'goTermsWordDescription.pickle', "rb"))

# parent_dict = pickle.load(open ( work_dir+"goTermsParent.pickle", "rb") )
children_dict = pickle.load(open ( work_dir+"goTermsChildren.pickle", "rb") )

GO_all_info_path = pickle.load(open ( work_dir+where_go_info+'/GO_all_info_go_graph.pickle',"rb")) # GO_all_info_go_graph

random.seed(a=2019)
fout = open("BERT_go_branch.txt","w")
for go, value in tqdm(GO_all_info_path.items()):
  if go not in children_dict: ## leaf node
    ## add in all ancestors

    if go not in go_def: 
      continue
    d1 = " ".join(s for s in go_def[go])
    d1 = re.sub(r"\t"," ",d1)
    d1 = re.sub(r"\n"," ",d1)
    sent = d1.lower()

    stack = [go] ## start with this leaf 
    while len(stack) > 0:

      if GO_all_info_path[stack[0]].parent_info is None: ## no parents 
        sent = sent + "\n\n" ## new document has 1 blank line
        break

      ## get one parent of leaf 
      parent = [p for p in GO_all_info_path[stack[0]].parent_info['neighbor_name'] if p in go_def]
      parent = random.choice(parent) # select 1 parent

      d2 = " ".join(s for s in go_def[parent])
      d2 = re.sub(r"\t"," ",d2)
      d2 = re.sub(r"\n"," ",d2)
        
      sent = sent + "\n" + d2.lower() ## one sent per line
      stack[0] = parent ## prepare to get parent of the current parent node

    fout.write(sent) 


fout.close()

