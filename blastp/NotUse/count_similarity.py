import pandas as pd
import numpy as np 
import os,sys,re,pickle

os.chdir('/u/flashscratch/d/datduong/goAndGeneAnnotationDec2018/')
where_train = "ecoli_yeast_human/full/redo_dec17_w2v17" 

add_name = "_metaGO" 


df_blast = pd.read_csv ( where_train+"/blast_testset"+add_name+".out" , header=None,skip_blank_lines=True)

## loop over each proteins (because some prot may match to a lot of stuffs)
prot = set ( list ( df_blast[0] ) ) 
print ('num gene from blast {}'.format(len(prot)) )

quantiles = df_blast[2].describe() ## similarity 
# np.where (df_blast[2] == np.max(df_blast[2]))
## quantiles
quantiles[4:7]


for add_name in [ "_metaGO" ] :  #, "_our"
  print (add_name)
  df_psiblast = pd.read_csv (  where_train+"/psiblast_testset"+add_name+".out" , header=None,skip_blank_lines=True)
  df_psiblast = df_psiblast.dropna()
  gene_name_test = list ( set (list ( df_psiblast[0] ) ) )
  try: 
    gene_name_test.remove('Search has CONVERGED!')
  except: 
    pass
  q = np.arange(0,.99,.05)
  quantiles_all = np.zeros( (len(gene_name_test),len(q) ) )
  counter = 0 
  for prot in gene_name_test: 
    df_p = df_psiblast [ df_psiblast[0] == prot ]
    quantiles = df_p[2].describe(percentiles=q) ## similarity 
    # print (quantiles)
    quantiles_all[counter] = quantiles [4:(len(q)+4)]
    counter = counter + 1  
  # np.where (df_blast[2] == np.max(df_blast[2]))
  ## quantiles
  print ( np.mean (quantiles_all, axis=0) )




z = pickle.load(open(where_train+"/seq-seq-prediction_metaGO.pd.pickle","rb"))
y = pickle.load(open(where_train+"/seq-seq-prediction_metaGO.pickle","rb"))

w = pickle.load(open("/u/flashscratch/d/datduong/goAndGeneAnnotationDec2018/benchmark_cofactor_method/1079.testing/gold_standard/test_gene_annot.pickle","rb"))


