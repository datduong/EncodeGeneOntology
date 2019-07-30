
## it seems GCN score is correlated to number of children/parents 

from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata, string, re, sys, os, pickle, pickle, gzip

import numpy as np
import pandas as pd 

from copy import deepcopy

import networkx
import obonet

## using the COO format of sklearn 

## deepgo/data/train

# work_dir = '/u/flashscratch/d/datduong/goAndGeneAnnotationDec2018/'
# work_dir = '/u/flashscratch/d/datduong/goAndGeneAnnotation/'
work_dir = '/u/flashscratch/d/datduong/deepgo/data/'

os.chdir (work_dir)

# Read the taxrank ontology
graph = obonet.read_obo('go.obo') # https://github.com/dhimmel/obonet
# graph.degree('GO:0007186')

input_score = '/u/scratch/d/datduong/geneOrtholog/MouseFlyScore/qnliFormatData17/GcnRelu300Cosine/score.9900.txt'
# index question  sentence  go1 go2 label score
df = pd.read_csv(input_score,sep="\t")
df = df.drop(columns="index")
df = df.drop_duplicates()

fout = open('/u/scratch/d/datduong/geneOrtholog/MouseFlyScore/qnliFormatData17/GcnRelu300Cosine/score.9900.degree.txt','w')
fout.write ('go1\tgo2\tlabel\tscore\tdegree1\tdegree2\tMeanDegree') 
for index,row in df.iterrows(): 
  d1 = graph.degree (row['go1']) 
  d2 = graph.degree (row['go2']) 
  try: 
    d3 = (d1+d2*1.0) / 2 ## stupid int 
  except: 
    continue
  fout.write ( "\n"+"\t".join( str(row[k]) for k in ['go1', 'go2', 'label', 'score'] ) + "\t" +str(d1) + "\t"+str(d2)+ "\t"+str(d3) )


fout.close() 

