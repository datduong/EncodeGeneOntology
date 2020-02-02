import pandas as pd 
import numpy as np 
import re,os,sys,pickle 

## some strange behavior 
# 0000009  0050412     neutral   test
# 0000009  0050412     neutral  train
# 0001054  0003899  entailment
# 0001055  0003899     neutral ***
# 0001055  0043842     neutral
# 0001055  0003899  entailment ***


def check_quality(df): 
  print (df.shape)
  try:
    df = df.drop(columns=['3']) ## pair must appear only once (one in each dataset)
    df = df.drop_duplicates()
    print(df.shape)  
  except: 
    pass
  df = df.drop(columns=['2']) ## remove label, pair must have unique label
  df = df.drop_duplicates()
  print(df.shape) 


main_dir = '/u/flashscratch/d/datduong/goAndGeneAnnotationDec2018/entailment_data/AicScore'
os.chdir(main_dir)

df = pd.read_csv("all_go_id_b2b.txt",sep="\t",dtype=str)
df = df.sort_values(by=['0'])
df = df.reset_index(drop=True)   
check_quality(df)

df = df.drop(columns=['3']) ## pair must appear only once (one in each dataset)
print(df.shape)
where_dup = df.duplicated(keep='first') ## keep first occurance
where_dup = np.where(where_dup)[0]
df.drop(df.index[where_dup], inplace=True)
df = df.reset_index(drop=True) 
print(df.shape)

df2 = df.drop(columns=['2']) ## remove label, pair must have unique label
where_dup = df2.duplicated(keep=False) ## remove both duplicates, because too confusing to keep which one
where_dup = np.where(where_dup)[0]
df.drop(df.index[where_dup], inplace=True)
df = df.reset_index(drop=True) 
print(df.shape)



df.to_csv("all_go_id_b2b_correct.txt",index=False,sep="\t")

## read in and shuffle ? 
## spit data 80 10 10 

df = pd.read_csv("all_go_id_b2b_correct.txt",sep="\t",dtype=str)
check_quality(df)

def train_validate_test_split(df, train_percent=.85, validate_percent=.05, seed=1234):
  np.random.seed(seed)
  perm = np.random.permutation(df.index)
  m = len(df.index)
  train_end = int(train_percent * m)
  validate_end = int(validate_percent * m) + train_end

  train = df.loc[perm[:train_end]]
  train['index'] = np.arange(0,train.shape[0]) # reset index

  validate = df.loc[perm[train_end:validate_end]]
  validate['index'] = np.arange(0,validate.shape[0])

  test = df.loc[perm[validate_end:]]
  test['index'] = np.arange(0,test.shape[0])

  return train, validate, test


train, validate, test = train_validate_test_split(df, seed=1234)
train.to_csv('train_go_id_b2b_correct.tsv',index=None,sep='\t')
validate.to_csv('dev_go_id_b2b_correct.tsv',index=None,sep='\t')
test.to_csv('test_go_id_b2b_correct.tsv',index=None,sep='\t')

