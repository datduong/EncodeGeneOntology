
import re,sys,os,pickle
import pandas as pd
import numpy as np

#! count all labels. in their original train + test file (which actually is just uniprot file)

os.chdir ('/u/scratch/d/datduong/deepgo/data/train')


# col = Entry Gene ontology IDs Sequence  Prot Emb

for onto in ['mf','cc','bp']:

  global_count_in_onto = {} #! global count

  for data_type in ['train','test']:
    fin = pd.read_csv( data_type+'-'+onto+'.tsv', sep='\t' )
    for index, row in fin.iterrows():
      label = row['Gene ontology IDs'].strip().split(';')
      for lab in label:
        if lab in global_count_in_onto:
          global_count_in_onto[lab] = global_count_in_onto[lab] + 1
        else:
          global_count_in_onto[lab] = 1

  #! order by alphabet
  fout = open( 'global-count-'+onto+'.tsv','w')
  terms = sorted ( list ( global_count_in_onto.keys() ) )
  for term in terms: # label \t count
    fout.write( term + '\t' + str(global_count_in_onto[term]) + '\n' )

  fout.close()



