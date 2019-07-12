

import pandas as pd 
import os,sys,re,pickle,gzip 
import numpy as np 
from tqdm import tqdm 

import networkx
import obonet

root = ['GO:0008372','GO:0005575','GO:0008150','GO:0000004', 'GO:0007582', 'GO:0044699', 'GO:0003674',' GO:0005554']

main_dir = '/u/scratch/d/datduong/goAndGeneAnnotationDec2018/ecoli_yeast_human'
os.chdir(main_dir)

# Read the taxrank ontology
graph = obonet.read_obo('../go.obo') # https://github.com/dhimmel/obonet


all_name_array = pd.read_csv("../go_name_in_obo.csv", header=None)
all_name_array = list (all_name_array[0])
# all_name_array = [ re.sub(r"GO:","",g) for g in all_name_array ] ## don't use GO: in the input files

label_map = {label : i for i, label in enumerate(all_name_array)}

df = pd.read_csv("ecoli_yeast_human_seq_go.txt",sep="\t")
print ('before filter len {}'.format(df.shape))

df['Sequence'] = df['Sequence'].astype('str')
max(df['Sequence'].str.len())

mask = (df['Sequence'].str.len() <= 2000) & (df['Sequence'].str.len() >= 20)
df = df.loc[mask]

print ('after filter len {}'.format(df.shape))

## check if all GO term was used in GCN or has def
## use the @go_name_in_obo
go_array = list ( df['Gene ontology IDs'] )

for counter, go in tqdm(enumerate(go_array)) : 
  go = go.split(";") ## split set to array 
  go = [g for g in go if g in label_map] # filter ?
  ## append parents 
  parents = []
  for g in go: 
    parents = parents + list ( networkx.descendants(graph, g) )
  go = parents + go 
  # go = [ re.sub(r"GO:","",g) for g in go ] ## don't use GO: in the input files
  go = sorted ( list ( set (go) ) )
  go = [g for g in go if g not in root] ## remove root 
  go = ";".join(g for g in go)
  go_array[counter] = go



df['Gene ontology IDs'] = go_array ## for sure will have definitions 

df.to_csv("ecoli_yeast_human_seq_go_is_a_obo_L20_2000.txt",index=False,sep="\t")

