

import numpy as np
import pandas as pd 
import os,sys,re,pickle


os.chdir ("/u/flashscratch/d/datduong/goAndGeneAnnotationDec2018/ecoli_yeast_human/full/redo_dec17_w2v17")
# file_name = '/u/flashscratch/d/datduong/goAndGeneAnnotationDec2018/ecoli_yeast_human/ecoli_yeast_human_seq_go_full_is_a_match_species.txt'
file_name = '/u/flashscratch/d/datduong/goAndGeneAnnotationDec2018/benchmark_cofactor_method/1079.testing/seq_tab_format.txt'
where_save = "/u/flashscratch/d/datduong/goAndGeneAnnotationDec2018/benchmark_cofactor_method/1079.testing/all_seq_pssm.pickle"

out_folder = 'per_seq_pssm'

df = pd.read_csv(file_name,dtype=str,sep="\t") ## "ArabidopsisUniprot.tab"
df = df.dropna()
gene_names = list (df['Entry'])

pssm_dict = {}

for g in gene_names: 
  pssm = None 
  header = [] 
  counter = -1 
  if not os.path.exists( out_folder+"/"+g+".pssm" ): 
    continue
  fin = open ( out_folder+"/"+g+".pssm" ,"r")
  for line in fin: 
    counter = counter + 1 
    line = line.strip().split() 
    if len(line)==0: 
      continue
    if line[0] == 'K': 
      break 
    if counter == 2: ## header 
      header = line
      continue
    line = line[2:(len(line)-2)] ## skip first 2, and last 2 
    if counter > 2: 
      if pssm is None: 
        pssm = np.array (line)
      else: 
        pssm = np.vstack ((pssm,line))
  #
  fin.close() 

  if pssm is not None:  
    pssm_dict[g] = pssm[:, 0:20] 



pickle.dump ( pssm_dict, open(where_save,'wb'))



