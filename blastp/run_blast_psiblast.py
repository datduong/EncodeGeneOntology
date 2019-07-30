
import os, sys, re, pickle, gzip
import pandas as pd
import numpy as np

## have to run psiblast to get seq. similarity

## then we process the file.

def score_go_term ( this_gene, score, gene_annot, go_score_array ): ##
  go_array = gene_annot [this_gene] ## go-array
  for g in go_array:
    if g not in go_score_array:
      go_score_array [g] = [score] # @score is the blast/psiblast score
    else:
      go_score_array [g].append(score)
  return go_score_array


def tally_over_n_template (df, gene_annot):

  weight_w = np.max( df[2] )/100
  go_score_array = {}
  score_over_all_n = np.sum ( df[2] )
  df = df.reset_index(drop=True)
  row_iterator = df.iterrows() ## is it faster ??

  ## go over each gene, get the GO terms.
  for i, row in row_iterator:
    go_score_array = score_go_term ( row[1], row[2], gene_annot, go_score_array ) ##

  ## now do the "summation"
  for g in go_score_array:
    score = go_score_array[g]
    score = np.sum(score)/ score_over_all_n
    go_score_array[g] = score ## each go will have "confident" score

  return go_score_array, weight_w



def submitJobs (where_train, what_set, where_blast, add_name) :
  ## blast and psi-blast will have the same format.

  if add_name == 'none':
    add_name = ""

  os.chdir('/u/flashscratch/d/datduong/goAndGeneAnnotationDec2018/')

  gene_annot = pickle.load ( open (where_train+'/train_gene_annot.pickle','rb') ) # ecoli_yeast_human/full

  ## @where_train because we save the file in the training location
  df_psiblast = pd.read_csv (  where_blast+"/psiblast_"+what_set+add_name+".out" , header=None,skip_blank_lines=True)
  df_psiblast = df_psiblast.dropna()
  df_psiblast = df_psiblast.reset_index(drop=True)

  gene_name_test = list ( set (list ( df_psiblast[0] ) ) )
  gene_name_test.sort()
  print ('\nnum of genes to be tested {}\n'.format(len(gene_name_test)))

  df_blast = pd.read_csv ( where_blast+"/blast_"+what_set+add_name+".out" , header=None,skip_blank_lines=True)

  prediction = {}

  for this_gene in gene_name_test :

    df_psiblast_g = df_psiblast[ df_psiblast[0] == this_gene ]
    df_psiblast_g = df_psiblast_g[ df_psiblast_g[1] != this_gene ] ## don't compare to self

    df_blast_g = df_blast[ df_blast[0] == this_gene ]
    df_blast_g = df_blast_g[ df_blast_g[1] != this_gene ] ## don't compare to self

    psiblast_go_score_array, w_psiblast = tally_over_n_template ( df_psiblast_g, gene_annot )
    blast_go_score_array, w_blast = tally_over_n_template ( df_blast_g, gene_annot )

    final_score = {}
    psiblast_go = list ( psiblast_go_score_array.keys() )
    blast_go = list ( blast_go_score_array.keys() )

    go_found = set ( psiblast_go + blast_go )
    if len(go_found) == 0:
      print ('not found any go term ?? {}'.format(this_gene))
      final_score[this_gene] = None
      continue

    for g in go_found:
      if (g in psiblast_go_score_array) and (g in blast_go_score_array) :
        x1 = psiblast_go_score_array[g] * (1-w_psiblast) + blast_go_score_array[g] * (w_psiblast)
      if (g in psiblast_go_score_array) and (g not in blast_go_score_array) :
        x1 = psiblast_go_score_array[g]
      if (g not in psiblast_go_score_array) and (g in blast_go_score_array) :
        x1 = blast_go_score_array[g]

      final_score[g] = x1

    prediction [this_gene] = final_score


  pickle.dump ( prediction, open(where_train+"/seq_seq_predict_go_"+what_set+add_name+".pickle","wb") )



if len(sys.argv)<1: ## run script
	print("Usage: \n")
	sys.exit(1)
else:
	submitJobs ( sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]  )









