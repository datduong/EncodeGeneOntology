import pickle, os, sys, re
import pandas as pd
import numpy as np

os.chdir('/u/scratch/d/datduong/deepgo/dataExpandGoSet16Jan2020/train/fold_1')
## >name
## seq

for onto in ['bp','cc','mf']: 
  for data_type in ['train', 'dev', 'test']: 
    df = pd.read_csv(data_type+'-'+onto+'-same-origin.tsv',sep='\t') # Entry Gene ontology IDs Sequence  Prot Emb
    fout = open( data_type+'-'+onto+'-same-origin.fasta', 'w' )
    for index,row in df.iterrows(): 
      fout.write(">"+row['Entry']+"\n"+row['Sequence']+"\n") 
    fout.close() 


