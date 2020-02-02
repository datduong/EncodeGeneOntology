

from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string, re, sys, os, pickle, gzip
from tqdm import tqdm
import numpy as np
import pandas as pd
from scipy import stats

## count #go for each gene, see why we see drop in performance for less known species

import compare_set

def CountGoPerGene (df,colname,gaf,ic):
  average = []
  go_seen = []
  ic_val = np.arange(0.1,.9,.1) * 0

  counter = 0.0

  for index, row in df.iterrows():

    if row['gene1'] in gaf:
      average.append ( len ( gaf[row['gene1']] ) ) ## num go per gene
      go_seen = go_seen + gaf[row['gene1']] ## we will keep the duplicated count

    if row['gene2'] in gaf:
      average.append ( len ( gaf[row['gene2']] ) ) ## num go per gene
      go_seen = go_seen + gaf[row['gene2']] ## we will keep the duplicated count


  ## look up Info Content value
  # go_seen = list ( set(go_seen) )
  ic_val = [ ic[go] for go in go_seen if go in ic ]
  print ('\ninfo content val')
  print ( stats.describe(ic_val) )
  print ( np.quantile(ic_val, np.arange(0.1,.9,.1)))

  return np.quantile(ic_val, np.arange(0.1,.9,.1))


pair_dir = '/u/scratch/d/datduong/'
gaf_dir = '/u/scratch/d/datduong/goAndGeneAnnotationMar2017/gafData2017/'

MapName = {'Human':'human','Fly':'fb','Worm':'wb','Yeast':'sgd','Mouse':'mgi'}

ic_per_gene = []

pairs = ['Yeast'] # 'Human',

for p in pairs:

  p = p.split()

  print ('\n\n{}'.format(p))

  spec1 = p[0]
  gaf1 = gaf_dir+MapName[spec1]+'_not_IEA.tsv'

  ic_combine = pickle.load(open(pair_dir+'goAndGeneAnnotationMar2017/'+spec1+'IC3ontology.pickle','rb'))

  df = pd.read_csv(pair_dir+spec1+'PPI3ontology/dataUsedInAuc.csv',dtype=str,sep=' ')

  GeneGOdb1 = compare_set.gaf2dict(gaf1)

  ic1 = CountGoPerGene(df,'gene1',GeneGOdb1,ic_combine)

  ic_per_gene.append(ic1)


ic_per_gene = np.array(ic_per_gene)
print (ic_per_gene)

# np.savetxt('ic_per_gene_ppi.csv', ic_per_gene)


