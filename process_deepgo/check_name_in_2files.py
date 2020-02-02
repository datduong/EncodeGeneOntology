import re,sys,pickle,os
import numpy as np 
import pandas as pd 

forigin = pd.read_csv('/local/datdb/deepgo/data/train/fold_1/dev-mf.tsv',sep='\t')
foriginname = sorted ( list (forigin['Entry']) ) 
print (len(foriginname))

f2 = pd.read_csv('/local/datdb/deepgo/dataExpandGoSet/train/fold_1/dev-mf-same-origin.tsv',sep='\t')
f2name = sorted ( list (f2['Entry']) ) 
print (len(f2name))

f1 = pd.read_csv('/local/datdb/deepgo/dataExpandGoSet/train/fold_1/dev-mf.tsv',sep='\t')
f1name = sorted ( list (f1['Entry']) ) 
print (len(f1name))


