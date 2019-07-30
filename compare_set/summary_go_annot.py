

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

## count #go for each gene, see why we see drop in performance for less known species


# Read the ontology
graph = obonet.read_obo('/u/scratch/d/datduong/goAndGeneAnnotationMar2017/go.obo') # https://github.com/dhimmel/obonet


import compare_set

def CountGoPerGene (df,colname,gaf,ic):
  average = []
  go_seen = []
  degree = []
  ic_val = []

  counter = 0.0
  for gene in df[colname]:
    if gene in gaf:
      average.append ( len ( gaf[gene] ) ) ## num go per gene
      go_seen = go_seen + gaf[gene] ## we will keep the duplicated count
      degree = degree + [ graph.degree (go) for go in gaf[gene] ]
      ic_val = ic_val + [ ic[go] for go in gaf[gene] if go in ic ]

  ## look up Info Content value

  print ('\ninfo content val')
  print ( stats.describe(ic_val) )
  print ( np.quantile(ic_val, np.arange(0.1,.9,.1)) )

  ## look up degree

  print ('\ninfo on degree')
  print ( stats.describe(degree) )
  print ( np.quantile(degree, np.arange(0.1,.9,.1)) )

  return np.quantile(ic_val, np.arange(0.1,.9,.1)) , np.quantile(degree, np.arange(0.1,.9,.1))

  ##
  # print ('\ngo count summary')
  # print ( stats.describe(average) )
  # print ( np.quantile(average, np.arange(0.1,.9,.1)))


pair_dir = '/u/scratch/d/datduong/geneOrtholog/'
gaf_dir = '/u/scratch/d/datduong/goAndGeneAnnotationMar2017/gafData2017/'

MapName = {'Human':'human','Fly':'fb','Worm':'wb','Yeast':'sgd','Mouse':'mgi'}

ic_list = []
degree_list = []

pairs = ['Human Mouse', 'Human Fly', 'Mouse Fly', 'Fly Worm']

for p in pairs:

  p = p.split()

  print ('\n\n{}'.format(p))

  spec1 = p[0]
  gaf1 = gaf_dir+MapName[spec1]+'_not_IEA.tsv'
  # ic1 = ''

  spec2 = p[1]
  gaf2 = gaf_dir+MapName[spec2]+'_not_IEA.tsv'
  # ic2 = ''

  ic_combine = pickle.load(open(pair_dir+spec1+spec2+'Score/'+spec1+spec2+'IC3ontolgy.pickle','rb'))

  df = pd.read_csv(pair_dir+spec1+spec2+'Score/dataUsedInAuc.csv',dtype=str,sep=' ')

  GeneGOdb1 = compare_set.gaf2dict(gaf1)
  GeneGOdb2 = compare_set.gaf2dict(gaf2)

  ic1, degree1 = CountGoPerGene(df,'gene1',GeneGOdb1,ic_combine)
  ic2, degree2 = CountGoPerGene(df,'gene2',GeneGOdb2,ic_combine)

  ic_list.append(ic1)
  ic_list.append(ic2)

  degree_list.append(degree1)
  degree_list.append(degree2)


print ('\nic')
ic_list = np.array(ic_list)
print (ic_list)


print ('\ndegree')
degree_list = np.array(degree_list)
print (degree_list)


np.savetxt('ic_per_gene_orthologs.csv', ic_list)
np.savetxt('degree_gene_orthologs.csv', degree_list)



