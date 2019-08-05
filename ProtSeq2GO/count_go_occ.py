


## count how many times a go term occur in the data 

import re,os,sys,pickle
import numpy as np 
import pandas as pd


main_dir = '/u/scratch/d/datduong/deepgo/data/train/fold_1'
os.chdir( main_dir )


for category in ['cc','bp','mf']: 

  go_counter = {} 

  df = pd.read_csv("train-"+category+".tsv",sep="\t")

  for counter, row in df.iterrows(): # Entry Gene ontology IDs Sequence  Prot Emb
    go = row['Gene ontology IDs'].strip().split(";")
    for g in go: 
      if g in go_counter: 
        go_counter[g] = go_counter[g] + 1
      else: 
        go_counter[g] = 1

  ## write go counter to file to plot 
  fout = open("CountGoInTrain-"+category+".tsv",'w')
  fout.write("GO\tcount\n")
  for k,v in go_counter.items(): 
    fout.write(k+"\t"+str(v)+"\n")

  fout.close() 

  pickle.dump(go_counter, open("CountGoInTrain-"+category+".pickle","wb"))


## 