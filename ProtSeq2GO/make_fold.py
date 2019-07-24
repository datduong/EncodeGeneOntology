

import pandas as pd 
import os,sys,re,pickle,gzip 
import numpy as np 
from tqdm import tqdm 

main_dir = '/u/scratch/d/datduong/goAndGeneAnnotationDec2018/ecoli_yeast_human'
os.chdir(main_dir)


## note: the ratio are computed to match the baseline ratio
def train_validate_test_split(df, train_percent=.71, validate_percent=.14, seed=1234):
  np.random.seed(seed)
  perm = np.random.permutation(df.index)
  m = len(df.index)
  train_end = int(train_percent * m)
  validate_end = int(validate_percent * m) + train_end

  train = df.ix[perm[:train_end]]
  # train['index'] = np.arange(0,train.shape[0]) # reset index

  validate = df.ix[perm[train_end:validate_end]]
  # validate['index'] = np.arange(0,validate.shape[0])

  test = df.ix[perm[validate_end:]]
  # test['index'] = np.arange(0,test.shape[0])

  return train, validate, test


all_name_array = pd.read_csv("../go_name_in_obo.csv", header=None)
all_name_array = list (all_name_array[0])
# all_name_array = [ re.sub(r"GO:","",g) for g in all_name_array ] ## don't use GO: in the input files

df = pd.read_csv("ecoli_yeast_human_seq_go_is_a_obo_L20_2000.txt",sep="\t")
df = df.dropna()

main_dir = main_dir + '/' + 'full_is_a_fold' 

if not os.path.exists(main_dir): 
  os.mkdir(main_dir)

for fold in [1,2]: 
  where_fold = os.path.join( main_dir , "fold_"+str(fold))
  if not os.path.exists(where_fold): 
    os.mkdir(where_fold)
  os.chdir(where_fold)
  train, validate, test = train_validate_test_split(df,train_percent=.905, validate_percent=.03095,seed=int(1234/fold))
  train.to_csv('train.tsv',index=None,sep="\t")
  validate.to_csv('dev.tsv',index=None,sep="\t")
  test.to_csv('test.tsv',index=None,sep="\t")



