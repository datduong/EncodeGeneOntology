


from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string, re, sys, os, pickle, gzip
from tqdm import tqdm
import numpy as np
import pandas as pd
from scipy import stats


import networkx
import obonet

import compare_set

def GetIcDegree (df,colname,gaf,ic):
  average = []
  go_seen = []
  degree = []
  ic_val = []

  counter = 0.0
  for index, row in tqdm(df.iterrows()):

    if row['gene1'] in gaf:
      average.append ( len ( gaf[row['gene1']] ) ) ## num go per gene
      ic_val = ic_val + [ ic[go] for go in gaf[row['gene1']] if go in ic ]

    if row['gene2'] in gaf:
      average.append ( len ( gaf[row['gene2']] ) ) ## num go per gene
      ic_val = ic_val + [ ic[go] for go in gaf[row['gene2']] if go in ic ]


  print ('\ninfo content val')
  print ( stats.describe(ic_val) )
  ic_val = np.quantile(ic_val, np.arange(0.1,.9,.1))
  print (ic_val)

  return ic_val


pair_dir = '/u/scratch/d/datduong/'
gaf_dir = '/u/scratch/d/datduong/goAndGeneAnnotationMar2017/gafData2017/'

ic_list = []
degree_list = []

pairs = ['Human Mouse', 'Human Fly', 'Mouse Fly', 'Fly Worm']

MapName = {'Human':'human','Fly':'fb','Worm':'wb','Yeast':'sgd','Mouse':'mgi'}

# Read the ontology
graph = obonet.read_obo('/u/scratch/d/datduong/goAndGeneAnnotationMar2017/go.obo') # https://github.com/dhimmel/obonet

## take positive/negative-pair, compare score with GO IC value

for p in ['Yeast','Human']:

  p = p.split()

  print ('\n\n{}'.format(p))

  spec1 = p[0]
  gaf1 = gaf_dir+MapName[spec1]+'_not_IEA.tsv'

  ic_combine = pickle.load(open(pair_dir+'goAndGeneAnnotationMar2017/'+spec1+'IC3ontology.pickle','rb'))

  where_data = '/u/scratch/d/datduong/'+spec1+'PPI3ontology/dataUsedInAuc.csv'

  GeneGOdb1 = compare_set.gaf2dict(gaf1) ## convert gene annotation into dict

  df = pd.read_csv( where_data, sep=" " )

  for label_type in [0,1]:

    df_label_type = df[df['label']==label_type] ## look at yes/no label only
    df_label_type = df_label_type.reset_index(drop=True)

    # what are the IC for terms found in each yes/no dataset

    ic_val1 = GetIcDegree(df_label_type,'gene1',GeneGOdb1, ic_combine) ## left and right hand side

    print ("\nlabel type {}".format(label_type))



