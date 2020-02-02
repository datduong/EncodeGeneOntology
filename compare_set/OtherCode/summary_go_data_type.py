


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

# Read the ontology
graph = obonet.read_obo('/u/scratch/d/datduong/goAndGeneAnnotationMar2017/go.obo') # https://github.com/dhimmel/obonet

def GetIcDegree (df,colname,gaf,ic):
  average = []
  go_seen = []
  degree = []
  ic_val = []

  counter = 0.0
  for gene in df[colname]:
    if gene in gaf:
      average.append ( len ( gaf[gene] ) ) ## num go per gene
      go_seen = go_seen + gaf[gene] ## we will keep the duplicated count
      # ic_val = ic_val + [ ic[go] for go in gaf[gene] if go in ic ]
      # degree = degree + [ graph.degree (go) for go in gaf[gene] if go in ic ]

  go_seen = list (set (go_seen))
  ic_val = [ ic[go] for go in go_seen if go in ic ]
  degree = [ graph.degree (go) for go in go_seen if go in ic ]

  print ('\ninfo content val')
  print ( stats.describe(ic_val) )
  ic_val = np.quantile(ic_val, np.arange(0.1,.9,.1)) 
  print (ic_val)

  print ('\ndegree val')
  print ( stats.describe(degree) )
  degree = np.quantile(degree, np.arange(0.1,.9,.1)) 
  print (degree)

  return ic_val , degree


pair_dir = '/u/scratch/d/datduong/geneOrtholog/'
gaf_dir = '/u/scratch/d/datduong/goAndGeneAnnotationMar2017/gafData2017/'

ic_list = []
degree_list = []

pairs = ['Human Mouse', 'Human Fly', 'Mouse Fly', 'Fly Worm']

MapName = {'Human':'human','Fly':'fb','Worm':'wb','Yeast':'sgd','Mouse':'mgi'}


## take positive/negative-pair, compare score with GO IC value 

for p in pairs: 

  p = p.split()

  print ('\n\n{}'.format(p))

  spec1 = p[0]
  gaf1 = gaf_dir+MapName[spec1]+'_not_IEA.tsv'

  spec2 = p[1]
  gaf2 = gaf_dir+MapName[spec2]+'_not_IEA.tsv'

  ic_combine = pickle.load(open(pair_dir+spec1+spec2+'Score/'+spec1+spec2+'IC3ontolgy.pickle','rb'))

  where_data = '/u/scratch/d/datduong/geneOrtholog/'+spec1+spec2+'Score/dataUsedInAuc.csv'

  GeneGOdb1 = compare_set.gaf2dict(gaf1) ## convert gene annotation into dict
  GeneGOdb2 = compare_set.gaf2dict(gaf2)

  df = pd.read_csv( where_data, sep=" " ) 
 
  for label_type in [0,1]: 

    print ("\nlabel type {}".format(label_type))

    df_label_type = df[df['label']==label_type] ## look at yes/no label only
    df_label_type = df_label_type.reset_index(drop=True)

    # what are the IC for terms found in each yes/no dataset 

    ic_val1 = GetIcDegree(df_label_type,'gene1',GeneGOdb1, ic_combine)
    ic_val2 = GetIcDegree(df_label_type,'gene2',GeneGOdb2, ic_combine)




