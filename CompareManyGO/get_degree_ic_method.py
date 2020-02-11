
## it seems BERT score is correlated to number of children/parents

from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata, string, re, sys, os, pickle, pickle, gzip

import numpy as np
import pandas as pd

from copy import deepcopy

import networkx
import obonet

## using the COO format of sklearn

# Read the taxrank ontology
graph = obonet.read_obo('/u/scratch/d/datduong/goAndGeneAnnotationMar2017/go.obo') # https://github.com/dhimmel/obonet # graph.degree('GO:0007186')


work_dir = '/u/scratch/d/datduong/goAndGeneAnnotationMar2017/RandomGOAnalysis/AIC/'
os.chdir (work_dir)

for input_score in ['Fly_ParentChild_go_analysis_cc.tsv','Fly_ParentChild_go_analysis_mf.tsv','Fly_ParentChild_go_analysis_bp.tsv','Fly_random_go_analysis_cc.tsv','Fly_random_go_analysis_mf.tsv','Fly_random_go_analysis_bp.tsv']:
  #
  df = pd.read_csv(input_score,sep="\t") # go1 go2 score ic1 ic2 label type
  fout = open( re.sub( r".tsv", r'.degree.tsv' , input_score ) , 'w' )
  fout.write ('go1\tgo2\tic1\tic2\tlabel\ttype\tscore\tdegree1\tdegree2\tMeanDegree')
  for index,row in df.iterrows():
    d1 = graph.degree (row['go1'])
    d2 = graph.degree (row['go2'])
    try:
      d3 = (d1+d2*1.0) / 2 ## stupid int
    except:
      continue
    fout.write ( "\n"+"\t".join( str(row[k]) for k in ['go1', 'go2', 'ic1', 'ic2', 'label', 'type', 'score'] ) + "\t" +str(d1) + "\t"+str(d2)+ "\t"+str(d3) )
  #
  fout.close()



