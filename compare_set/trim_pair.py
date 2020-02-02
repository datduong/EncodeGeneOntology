
import os,re,sys,gzip
import pandas as pd 


#### extract only pairs we need to do, not all pairs in the orthologs set are needed
#### data came from previous studies in 2017 and 2018 

pair_name = 'HumanFly'

os.chdir('/u/scratch/d/datduong/geneOrtholog/'+pair_name+'Score/')

df = pd.read_csv('/u/scratch/d/datduong/geneOrtholog/'+pair_name+'Score/'+pair_name+'Scoreall3wSimDef.txt',sep=" ",header=None)
df = df [ [0,1,df.shape[1]-1] ] ## get these col only
df.columns = ['gene1','gene2','label']

df.to_csv(pair_name+'Ortholog2TestTrim.txt',sep=" ",index=None)


## we can use the same code above for prot interaction network within human or yeast
import os,re,sys,gzip
import pandas as pd 

pair_name = 'Human'

os.chdir('/u/scratch/d/datduong/'+pair_name+'PPI3ontology/')

df = pd.read_csv('/u/scratch/d/datduong/'+pair_name+'PPI3ontology/score'+pair_name+'PPIall3wSimDef.txt',sep=" ",header=None)
df = df [ [0,1,df.shape[1]-1] ] ## get these col only
df.columns = ['gene1','gene2','label']

df.to_csv(pair_name+'PPI2TestTrim.txt',sep=" ",index=None)


