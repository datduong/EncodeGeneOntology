
import numpy as np
import pandas as pd 
import os,sys,re,pickle,gzip

## original github gives only train/test set. 
## in their keras code, they split the train into 2 set. 

# split using int(n * 0.8) 

def train_validate_test_split(df, train_percent=.80, seed=1234):
  
  np.random.seed(seed)
  perm = np.random.permutation(df.index)
  m = len(df.index)

  train_end = int(train_percent * m)

  train = df.ix[perm[:train_end]]

  validate = df.ix[perm[train_end:]]

  return train, validate


main_dir = "/u/scratch/d/datduong/deepgo/data/train/"
os.chdir(main_dir)



for ontology in ['cc','mf','bp']: 
  
  ## original github gives only train/test set. 
  ## in their keras code, they split the train into 2 set. 
  ## we will do preprocessing split. take the provided train.tsv, split into some folds
  df = pd.read_csv(main_dir+'train-'+ontology+'.tsv',dtype=str) 

  for fold in [1,2,3,4,5]: 
    
    where_fold = os.path.join( main_dir , "fold_"+str(fold))
    if not os.path.exists(where_fold): 
      os.mkdir(where_fold)

    os.chdir(where_fold)

    train, validate = train_validate_test_split(df,seed=int(1234/fold))
    train.to_csv('train-'+ontology+'.tsv',index=None)
    validate.to_csv('dev-'+ontology+'.tsv',index=None)


