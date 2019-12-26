import pickle, os, sys, gzip, re
import pandas as pd
import numpy as np
from tqdm import tqdm

import gensim
from gensim.models import KeyedVectors

sys.path.append ("/u/flashscratch/d/datduong/GOmultitask/")
import ProtSeq2GO.evaluation_metric as evaluation_metric

import blastp.ExpandGOSet as ExpandGOSet

## can be sloppy , don't need a whole arg input.

def submitJobs (main_dir, data_dir, blast_result_dir, what_set, ontology_type, wordvec_file,wordvec_file_small,label_subset_file) :

  ## @blast_result_dir is where we output the blast score, this is not same as @data_dir... should change pipeline ? 

  if wordvec_file_small == "none": 
    ExpandGOSetOnBlast = ExpandGOSet.GOVector(wordvec_file,label_subset_file)
  else: 
    ExpandGOSetOnBlast = ExpandGOSet.GOVector(wordvec_file,label_subset_file,wordvec_file_small=wordvec_file_small,cut_point=0.95)

  BlastResult = pickle.load ( open(blast_result_dir+what_set+"-"+ontology_type+".dict.pickle","rb") )

  print ('number of prot with prediction from blast-style {}'.format(len(BlastResult))) 
  
  BlastResultExpand = ExpandGOSetOnBlast.ExpandBlast(BlastResult)

  pickle.dump ( BlastResultExpand, open(blast_result_dir+what_set+"-"+ontology_type+".expand.pickle","wb") )

  df = pd.read_csv(data_dir+what_set+"-"+ontology_type+".tsv",sep="\t")
  prot_array = list(df['Entry']) ## only need name to retain the same ordering
  prediction = ExpandGOSetOnBlast.dict2matrix(BlastResultExpand,prot_array)
  print (prediction)

  ## get true label
  true_label = pickle.load ( open(data_dir+what_set+"-"+ontology_type+".TrueLabel.pickle","rb") )
  print ('number of prot in test set {}'.format(len(true_label)))
  true_label = ExpandGOSetOnBlast.truelabel2matrix(true_label,prot_array)
  print (true_label)

  print ('remove prot that blast did not find')
  get_found = np.where ( np.sum(prediction,1) > 0 ) [ 0 ]
  prediction = prediction[get_found,:]
  true_label = true_label[get_found,:]

  metric = evaluation_metric.all_metrics ( np.round(prediction), true_label, yhat_raw=prediction, k=15 )
  evaluation_metric.print_metrics( metric )

  ## !!!! 
  print ('\n\nnot expand GO set')
  prediction = ExpandGOSetOnBlast.dict2matrix(BlastResult,prot_array)
  get_found = np.where ( np.sum(prediction,1) > 0 ) [ 0 ]
  prediction = prediction[get_found,:]
  metric = evaluation_metric.all_metrics ( np.round(prediction), true_label, yhat_raw=prediction, k=15 )
  evaluation_metric.print_metrics( metric )

  print ('\n\nsee example')
  print ( BlastResult['Q92543'] ) 
  print ( BlastResultExpand['Q92543'] ) 

if len(sys.argv)<1: ## run script
	print("Usage: \n")
	sys.exit(1)
else:
	submitJobs ( sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7], sys.argv[8] )



