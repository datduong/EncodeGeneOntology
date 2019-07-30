
# must do this because each query has only 1 seq

import pandas as pd 
import os,sys,re

os.chdir ("/u/flashscratch/d/datduong/goAndGeneAnnotationDec2018/ecoli_yeast_human/full/redo_dec17_w2v17")
# file_name = '/u/flashscratch/d/datduong/goAndGeneAnnotationDec2018/ecoli_yeast_human/ecoli_yeast_human_seq_go_full_is_a_match_species.txt'
file_name = '/u/flashscratch/d/datduong/goAndGeneAnnotationDec2018/benchmark_cofactor_method/1079.testing/seq_tab_format.txt'
out_folder = 'per_seq_fasta'


df = pd.read_csv(file_name,dtype=str,sep="\t") ## "ArabidopsisUniprot.tab"
df = df.dropna()
row_iterator = df.iterrows() ## is it faster ?? 
for i, row in row_iterator:
  gene_name = row['Entry']
  fout = open(out_folder+'/'+gene_name+".fasta","w")
  fout.write (">"+row['Entry']+"\n"+row['Sequence']+"\n")
  fout.close() 


A1A546