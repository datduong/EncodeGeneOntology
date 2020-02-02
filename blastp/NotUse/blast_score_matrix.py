
import pickle, os, sys, gzip, re
import pandas as pd
import numpy as np

sys.path.append ("/u/scratch/d/datduong/GOmultitask")
import ProtSeq2GO.evaluation_metric as evaluation_metric


def submitJobs (main_dir,where_train,set_type,where_test,label_subset_file,add_name): 

  os.chdir ( main_dir )

  ground_truth_annot = pickle.load(open(where_test,"rb"))
  print ('num of gene in ground truth {}'.format(len(ground_truth_annot)))
  genes = sorted ( list (ground_truth_annot.keys()) ) ## alphabet, actually doesn't matter. just have to be consistent

  print ("\n\nload prediction pickle")
  prediction = pickle.load ( open(where_train+"/"+set_type+"-"+add_name+".dict.pickle","rb") )

  ## for each gene, fill in the prediction matrix
  df = pd.read_csv(label_subset_file, header=None)
  label_to_test = sorted ( list (df[0]) )
  label_index_map = {val:number for number, val in enumerate(label_to_test)} ## will make index look up very fast

  print ('convert prediction pickle into matrix form')
  prediction_np = np.zeros( (len(genes), len(label_index_map)) )

  for g in genes :
    if g not in prediction:
      continue
    go_assign = list ( prediction[g].keys() )
    go_assign.sort()
    score = [prediction[g][go] for go in go_assign]
    location = [label_index_map[go] for go in go_assign]
    ## assign the score
    prediction_np [ genes.index(g), location ] = score

  ## convert np into pd to get row names, some what stupid, because we only need the matrix, but let's be safe. 
  df = pd.DataFrame(prediction_np, index=genes)
  pickle.dump ( df, open(where_train+"/"+set_type+"-"+add_name+".pd.pickle","wb"))

  ## filter out to only go terms in training set
  truth_np = np.zeros( (len(genes), len(label_index_map)) )

  for g in genes :
    go_assign = sorted ( ground_truth_annot[g][0].strip().split(";") ) 
    location = [label_index_map[go] for go in go_assign if go in label_index_map ] ## !! record only GO we saw in training
    truth_np [ genes.index(g), location ] = 1 ## assign the score


  print ('\n\naccuracy meta-go\n\n')
  animo_go_metric = evaluation_metric.all_metrics ( np.round(prediction_np), truth_np, yhat_raw=prediction_np, k=15 ) ##  [ 0:(16*3) , :]
  evaluation_metric.print_metrics( animo_go_metric )



if len(sys.argv)<1: ## run script
	print("Usage: \n")
	sys.exit(1)
else:
	submitJobs ( sys.argv[1] , sys.argv[2] , sys.argv[3] , sys.argv[4], sys.argv[5], sys.argv[6])



