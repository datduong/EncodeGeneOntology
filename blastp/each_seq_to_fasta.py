

import os,sys,re,pickle
import pandas as pd
import numpy as np

os.chdir ("/u/scratch/d/datduong/deepgo/data/train/fold_1")

for category in ['bp','mf','cc']:
  for file_type in ['train','test','dev']:

    ground_truth = {} ## create a dictionary of ground trut
    fout = open(file_type+"-"+category+".fasta","w")
    
    df = pd.read_csv(file_type+"-"+category+".tsv",dtype=str,sep="\t") # Entry Gene ontology IDs Sequence  Prot Emb
    df = df.dropna()

    row_iterator = df.iterrows() ## is it faster ??
    for i, row in row_iterator:
      ground_truth[ row['Entry'] ] = row['Gene ontology IDs'].strip().split(";")
      fout.write (">"+row['Entry']+"\n"+row['Sequence']+"\n")

    pickle.dump ( ground_truth, open(file_type+"-"+category+".TrueLabel.pickle","wb"))
    fout.close()


