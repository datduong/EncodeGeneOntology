
import os, sys, re, pickle, gzip
import pandas as pd
import numpy as np
from tqdm import tqdm

import fmax
import evaluation_metric

## have to run psiblast to get seq. similarity
## then we process the file.

def MakeGroundTruthText2Dict (filename):
  prot_annot = {}
  df = pd.read_csv(filename,sep="\t")
  for index, row in df.iterrows():
    prot_annot [ row['Entry'] ] = row['Gene ontology IDs'].strip().split(';')
  return prot_annot, list(df['Entry']) ## return same ordering for name... notice do not sort protein name, we keep them as they appear in the tsv

def load_true_data (filename,label_lookup):
  # @label_lookup is lookup dict of all label tested
  true_label, prot_name = MakeGroundTruthText2Dict(filename)
  ground_truth = np.zeros([len(true_label),len(label_lookup)])
  for index,name in enumerate(prot_name):
    where1 = np.array ( [label_lookup[n] for n in true_label[name]] ) # lookup the column of labels
    ground_truth[index,where1] = 1
  return ground_truth, prot_name

def score_go_term ( this_prot, score, prot_annot, go_score_array ): ##
  go_array = prot_annot [this_prot] ## go-array
  for g in go_array:
    if g not in go_score_array:
      go_score_array [g] = [score] # @score is the blast/psiblast score
    else:
      go_score_array [g].append(score)
  return go_score_array

def tally_over_n_template (df, prot_annot):

  weight_w = np.max( df[2] )/100
  go_score_array = {}
  score_over_all_n = np.sum ( df[2] )
  df = df.reset_index(drop=True)
  row_iterator = df.iterrows() ## is it faster ??

  ## go over each prot, get the GO terms.
  for i, row in row_iterator:
    go_score_array = score_go_term ( row[1], row[2], prot_annot, go_score_array ) ##

  ## now do the "summation"
  for g in go_score_array:
    score = go_score_array[g]
    score = np.sum(score)/ score_over_all_n
    go_score_array[g] = score ## each go will have "confident" score

  return go_score_array, weight_w

def order_go_score (score,label_lookup): # @score is dict {go1:0.5, go2:0.4}
  out = np.zeros(len(label_lookup)) ## everything 0 first
  ## select position not 0
  sorted_score = sorted(list(score.keys())) ## must sort the keys of dict. all labels are sorted by alphabet
  where_found = np.array ( [ label_lookup[n] for n in sorted_score ] ) ## get back position that is the same for the alphabet ordering
  score_found = np.array ( [ score[n] for n in sorted_score ] ) ## get back score
  out[where_found] = score_found
  return out

def submitJobs (main_dir, data_dir, blast_result_dir, what_set, ontology_type, all_test_label,add_name='none') :

  if add_name=='none':
    add_name = ""

  #### blast and psi-blast will have the same format.
  ## @all_test_label is file of all labels to be tested, adding this so that we return a matrix num_ob x num_label
  os.chdir(main_dir)

  ## labels to be tested
  all_test_label = pd.read_csv(all_test_label,header=None)
  print ('\nsort labels to be tested, we do the same when using NN model.')
  all_test_label = sorted ( list(all_test_label[0]) )
  label_lookup = {value:index for index,value in enumerate(all_test_label)}

  ## prot annotation train set, will be used later to infer assignment in testset
  ## can only predict what is found in train set if we use blast
  print ('load go annotation for train data')
  ## we can convert text into dict on-the-fly
  # try:
  #   prot_annot = pickle.load ( open (data_dir+'train-'+ontology_type+'.TrueLabel.pickle','rb') )
  # except:
  # train-mf.tsv

  prot_annot, prot_name_train = MakeGroundTruthText2Dict(data_dir+'train-'+ontology_type+add_name+'.tsv')
  print ('\nnum of prots in train data {}\n'.format(len(prot_annot)))

  print ('load go annotation for test data')
  ## COMMENT get true labels
  ## COMMENT 'test-'+ontology_type+'.tsv' has different ordering than 'test-'+ontology_type+'-input.tsv'
  print ('test file name {}'.format(data_dir+'test-'+ontology_type+add_name+'.tsv')) ##!!##!!

  ground_truth, prot_name_test = load_true_data (data_dir+'test-'+ontology_type+add_name+'.tsv',label_lookup) ##!!##!!
  print ('\nnum of prots in test data {}\n'.format(len(prot_name_test)))

  print ('\nread psiblast result')
  df_psiblast = pd.read_csv ( blast_result_dir+what_set+"-"+ontology_type+".psiblast.txt" , header=None, skip_blank_lines=True )
  df_psiblast = df_psiblast.dropna()
  df_psiblast = df_psiblast.reset_index(drop=True)

  prot_name_in_psi = sorted ( list ( set (list ( df_psiblast[0] ) ) ) )
  print ('\nnum of prots from test found in psiblast {}, we may be unable to find match for all test sequence\n'.format(len(prot_name_in_psi)))

  print ('\nread blast result')
  df_blast = pd.read_csv ( blast_result_dir+what_set+"-"+ontology_type+".blast.txt" , header=None,skip_blank_lines=True )

  ## should make prediction as a matrix
  # prediction = {}
  prediction = np.zeros([len(prot_name_test),len(label_lookup)])

  in_psi = set(df_psiblast[0])
  in_blast = set(df_blast[0])

  for index,this_prot in tqdm(enumerate(prot_name_test)) :

    if (this_prot not in in_psi) and (this_prot not in in_blast):
      print ('not found in both blast and psiblast {}'.format(this_prot))
      continue

    df_psiblast_g = df_psiblast[ df_psiblast[0] == this_prot ]
    df_psiblast_g = df_psiblast_g[ df_psiblast_g[1] != this_prot ] ## don't compare to self

    df_blast_g = df_blast[ df_blast[0] == this_prot ]
    df_blast_g = df_blast_g[ df_blast_g[1] != this_prot ] ## don't compare to self

    psiblast_go_score_array, w_psiblast = tally_over_n_template ( df_psiblast_g, prot_annot )
    blast_go_score_array, _ = tally_over_n_template ( df_blast_g, prot_annot )

    final_score = {}
    psiblast_go = list ( psiblast_go_score_array.keys() )
    blast_go = list ( blast_go_score_array.keys() )

    go_found = set ( psiblast_go + blast_go )
    if len(go_found) == 0: ## funky stuffs ??
      print ('pass 1st screen in blast+psiblast but not found any go term ?? {}'.format(this_prot))
      final_score[this_prot] = None
      continue

    for g in go_found: ## average between psiblast and blast
      if (g in psiblast_go_score_array) and (g in blast_go_score_array) :
        x1 = psiblast_go_score_array[g] * (1-w_psiblast) + blast_go_score_array[g] * (w_psiblast)
      if (g in psiblast_go_score_array) and (g not in blast_go_score_array) :
        x1 = psiblast_go_score_array[g]
      if (g not in psiblast_go_score_array) and (g in blast_go_score_array) :
        x1 = blast_go_score_array[g]
      final_score[g] = x1 ## each GO term has a score for this one protein

    ## done with this one protein
    prediction [index] = order_go_score (final_score,label_lookup)

    ## filter down original set so things run faster
    # df[~df.countries.isin(countries)]
    # df_psiblast = df_psiblast[ ~df_psiblast[0].isin([this_prot]) ]
    # df_blast = df_blast[ ~df_blast[0].isin([this_prot]) ]

    # if index > 10:
    #   print (prediction[0:10])
    #   exit()

  ## finish all proteins

  pickle.dump ( {'prediction':prediction, 'true_label':ground_truth}, open(blast_result_dir+what_set+"-"+ontology_type+"-prediction.pickle","wb") )

  result = evaluation_metric.all_metrics ( np.round(prediction) , ground_truth, yhat_raw=prediction, k=[5,10,15,20,25]) ## we can pass vector of P@k and R@k
  evaluation_metric.print_metrics( result )



if len(sys.argv)<1: ## run script
	print("Usage: \n")
	sys.exit(1)
else:
	submitJobs ( sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7] )









