

## concat 

from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata, string, re, sys, os, pickle, pickle, gzip

import numpy as np
import pandas as pd

from copy import deepcopy

import networkx
import obonet


# encoder_setting = {'BiLSTM':'cosine.bilstm768','Onto2vec':'cosine.768', 'GCN':'cosine.768' } 
# encoder_shortname = {'BiLSTM':'BiLSTM','Onto2vec':'Onto2vec','GCN':'GCN'}
 
# encoder_setting = {'cosine.AveWordClsSep768.Linear768.Layer12': 'BERT' , 'cosine.Cls768.Linear768':'BERT' } ##
# encoder_shortname = {'cosine.AveWordClsSep768.Linear768.Layer12':'BERT','cosine.Cls768.Linear768':'BERT'}

# encoder_setting = {'BertGOName768': 'BertFineTuneGOEmb768'  } ##
# encoder_shortname = {'BertGOName768':'BertFineTuneGOEmb768' }

encoder_setting = {'cosine.1024': 'Elmo'  } ##
encoder_shortname = {'cosine.1024':'Elmo' }


for encoder_name in encoder_setting: 
  # if encoder_name == 'BertAveWordClsSepLayer12': 
  #   continue
  work_dir = '/local/datdb/goAndGeneAnnotationMar2017/RandomGOAnalysis/Elmo/'+encoder_name+'/' # +encoder_setting[encoder_name]
  os.chdir (work_dir)
  large_df = None
  for input_score in ['random_go_analysis_cc.'+encoder_shortname[encoder_name]+'.degree.txt', 'random_go_analysis_mf.'+encoder_shortname[encoder_name]+'.degree.txt', 'random_go_analysis_bp.'+encoder_shortname[encoder_name]+'.degree.txt' , 'ParentChild_go_analysis_bp.'+encoder_shortname[encoder_name]+'.degree.txt' , 'ParentChild_go_analysis_cc.'+encoder_shortname[encoder_name]+'.degree.txt', 'ParentChild_go_analysis_mf.'+encoder_shortname[encoder_name]+'.degree.txt']:
    #
    df = pd.read_csv(input_score,sep="\t") # index question  sentence  go1 go2 label score
    # df = df.drop(columns="index")
    df = df.drop_duplicates()
    if large_df is None: 
      large_df = deepcopy(df)
    else: 
      large_df = pd.concat([large_df,df],0)
  ## write out 
  # 
  large_df.to_csv("go_analysis_all.degree.txt",index=None,sep="\t") 



