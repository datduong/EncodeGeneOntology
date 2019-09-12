
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


# Read the taxrank ontology
graph = obonet.read_obo('/local/datdb/goAndGeneAnnotationMar2017/go.obo') # https://github.com/dhimmel/obonet
# graph.degree('GO:0007186')

# input_score = 'random_go_analysis_cc.GCN.txt'
# work_dir = '/local/datdb/goAndGeneAnnotationMar2017/RandomGOAnalysis/GCN/cosine.768.reduce300ClsVec'

# encoder_setting = {'Onto2vec':'cosine.768', 'GCN':'cosine.768' } 
# encoder_shortname = {'Onto2vec':'Onto2vec','GCN':'GCN'}


# encoder_setting = {'BiLSTM':'cosine.bilstm768'} 
# encoder_shortname = {'BiLSTM':'BiLSTM'}
# encoder_setting = {'BertGOName768': 'BertFineTuneGOEmb768'  } ##
# encoder_shortname = {'BertGOName768':'BertFineTuneGOEmb768' }

# encoder_setting = {'cosine.AveWordClsSep768.Linear768.Layer12': 'BERT' , 'cosine.Cls768.Linear768':'BERT' } ##
# encoder_shortname = {'cosine.AveWordClsSep768.Linear768.Layer12':'BERT','cosine.Cls768.Linear768':'BERT'}

encoder_setting = {'cosine.1024': 'Elmo'  } ##
encoder_shortname = {'cosine.1024':'Elmo' }


for encoder_name in encoder_setting: 
  # if encoder_name == 'BertAveWordClsSepLayer12': 
  #   continue
  work_dir = '/local/datdb/goAndGeneAnnotationMar2017/RandomGOAnalysis/Elmo/'+encoder_name+'/' #+encoder_setting[encoder_name]
  os.chdir (work_dir)
  for input_score in ['random_go_analysis_cc.'+encoder_shortname[encoder_name]+'.txt', 'random_go_analysis_mf.'+encoder_shortname[encoder_name]+'.txt', 'random_go_analysis_bp.'+encoder_shortname[encoder_name]+'.txt' , 'ParentChild_go_analysis_bp.'+encoder_shortname[encoder_name]+'.txt' , 'ParentChild_go_analysis_cc.'+encoder_shortname[encoder_name]+'.txt', 'ParentChild_go_analysis_mf.'+encoder_shortname[encoder_name]+'.txt']:
    #
    df = pd.read_csv(input_score,sep="\t") # index question  sentence  go1 go2 label score
    df = df.drop(columns="index")
    df = df.drop_duplicates()
    fout = open( re.sub( r".txt", r'.degree.txt' , input_score ) , 'w' )
    fout.write ('go1\tgo2\tlabel\ttype\tscore\tdegree1\tdegree2\tMeanDegree')
    for index,row in df.iterrows():
      d1 = graph.degree (row['go1'])
      d2 = graph.degree (row['go2'])
      try:
        d3 = (d1+d2*1.0) / 2 ## stupid int
      except:
        continue
      fout.write ( "\n"+"\t".join( str(row[k]) for k in ['go1', 'go2', 'label', 'type', 'score'] ) + "\t" +str(d1) + "\t"+str(d2)+ "\t"+str(d3) )
    #
    fout.close()

