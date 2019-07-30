
from __future__ import print_function, division
from io import open
import unicodedata
import string, re, sys, pickle, os, gzip
import numpy as np
import pandas as pd 

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# sys.path.append ("/u/flashscratch/d/datduong/GOmultitask")
# import helper 


## create batch based on pssm 
## these are can already be treated as the output of an emb 

os.chdir ("/u/flashscratch/d/datduong/goAndGeneAnnotationDec2018/ecoli_yeast_human/full/redo_dec17_w2v17")

pssm_dict = pickle.load ( open("/u/flashscratch/d/datduong/goAndGeneAnnotationDec2018/ecoli_yeast_human/full/redo_dec17_w2v17/all_seq_pssm.pickle",'rb'))


# os.chdir ("/u/flashscratch/d/datduong/goAndGeneAnnotationDec2018/benchmark_cofactor_method/1079.testing/gold_standard/redo_dec17_w2v17")

# pssm_dict = pickle.load ( open("/u/flashscratch/d/datduong/goAndGeneAnnotationDec2018/benchmark_cofactor_method/1079.testing/all_seq_pssm.pickle",'rb'))


for data_type in ['dev','train','test'] : #,'dev','train'

  batch = pickle.load(open(data_type+"_main_data_b8.pickle",'rb'))
  print ('\n\nexample before')
  print (batch[1])

  for b in batch: 
    genes = [g for g in batch[b]['gene_names'] if g in pssm_dict ]
    gene_names = batch[b]['gene_names'].tolist() 
    genes_wrt_whole = [gene_names.index(g) for g in genes ] ## index of genes found 
    max_len = np.max ( batch[b]['sequence_indexing_len'].data.numpy()[genes_wrt_whole] ) ## used for padding 

    feature = np.zeros( (len(genes), 20, max_len )) ## @20 is hardcoded by BLAST
    counter = 0
    for g in genes: 
      this_pssm = pssm_dict[g].transpose() 
      zeros = np.zeros((this_pssm.shape[0],max_len-this_pssm.shape[1]))
      feature[counter] = np.hstack ((this_pssm,zeros))
      counter = counter + 1 

    ## now assemble into batch 
    ## dict_keys(['input_sequence_indexing', 'sequence_indexing_len', 'true_label', 'gene_names'])
    batch[b]['true_label'] = batch[b]['true_label'][genes_wrt_whole] # smaller true data 
    batch[b]['sequence_indexing_len'] = batch[b]['sequence_indexing_len'][genes_wrt_whole] # smaller true data 
    batch[b]['gene_names'] = batch[b]['gene_names'][genes_wrt_whole] # smaller true data 
    batch[b]['input_sequence_indexing'] = torch.FloatTensor ( feature ) # will be 3D now


  #
  pickle.dump(batch,open(data_type+"_main_data_pssm_b8.pickle",'wb'))
  print ('example after')
  print (batch[1])







      

      


