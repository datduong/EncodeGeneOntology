
import pickle, os, sys, gzip, re
import pandas as pd 
import numpy as np 

#
sys.path.append ("/u/flashscratch/d/datduong/GOmultitask/")
import evaluation_metric
import fmax


# prediction = pickle.load ( open("/u/flashscratch/d/datduong/goAndGeneAnnotationDec2018/benchmark_cofactor_method/1079.testing/trainset_has_rare_go/seq-seq-prediction.pickle","rb") ) 

def submitJobs (where_train,set_type,where_test, add_name, do_split ): ## @do_split is needed if we use metaGO data 

  if add_name == 'none':
    add_name = "" 

  os.chdir ( '/u/flashscratch/d/datduong/goAndGeneAnnotationDec2018/')

  test_gene_annot = pickle.load(open(where_test+"/"+set_type+"_gene_annot.pickle","rb"))
  print ('num of gene to be tested {}'.format(len(test_gene_annot)))

  print ('\n\nmust use the prot names in the annot, not psiblast outcome\n\n')
  genes = list (test_gene_annot.keys())
  genes.sort() ## alphabet 

  prediction = pickle.load ( open(where_train+"/seq_seq_predict_go_"+add_name+".pickle","rb") ) 

  ## for each gene, fill in the prediction matrix 
  label_index_map = pickle.load ( open (where_train+"/label_index_map.pickle","rb") ) 
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

  ## convert np into pd to get row names 
  df = pd.DataFrame(prediction_np, index=genes)
  pickle.dump ( df, open(where_train+"/seq_seq_predict_go_"+add_name+".pd.pickle","wb"))

  ## filter out to only go terms in training set 
  truth_np = np.zeros( (len(genes), len(label_index_map)) )

  for g in genes : 
    if do_split == 1: 
      if ";" in test_gene_annot[g][0]: 
        go_assign = test_gene_annot[g][0].strip().split(";")
      else: 
        go_assign = test_gene_annot[g][0].strip().split(",")
    else: 
      go_assign = test_gene_annot[g]

    #
    go_assign.sort() 
    go_assign = [re.sub("GO:","",go) for go in go_assign]
  
    location = [label_index_map[go] for go in go_assign if go in label_index_map ] ## !! record only GO we saw in training 
    ## assign the score  
    truth_np [ genes.index(g), location ] = 1 
 
  print ('animo GO prediction')
  print (prediction_np)
  track_prec = []
  track_rec = []
  for k in [5,10,15,20,25,30,35,40]: 
    animo_go_metric = evaluation_metric.all_metrics ( np.round(prediction_np), truth_np, yhat_raw=prediction_np, k=k ) ##  [ 0:(16*3) , :]
    if k == 5 : 
      evaluation_metric.print_metrics( animo_go_metric )
    track_prec.append(animo_go_metric['prec_at_'+str(k)])
    track_rec.append(animo_go_metric['rec_at_'+str(k)])

  #
  fmax_val = fmax.f_max ( truth_np, prediction_np, threshold=np.arange(0,1,.02) )
  print ('fmax value {}'.format ( fmax_val ) ) 
  print ('precision/recall at K')
  print (track_prec)
  print (track_rec)


  label_bio_type = pickle.load( open( where_train+'/label_bio_type.pickle','rb') )
  # common30 = pickle.load ( open(where_train+"/common_index30.pickle","rb"))
  # label_bio_type['common30'] = common30

  for bio_type in label_bio_type: 
    index = label_bio_type [ bio_type ] 
    print ( "\n\n"+bio_type)
    print ( index[0:10] )
    track_prec = []
    track_rec = []
    for k in [5,10,15,20,25,30,35,40]: 
      animo_go_metric = evaluation_metric.all_metrics ( np.round(prediction_np[: , index]), truth_np[: , index], yhat_raw=prediction_np[: , index], k=k)
      if k == 5 : 
        evaluation_metric.print_metrics( animo_go_metric )
      track_prec.append(animo_go_metric['prec_at_'+str(k)])
      track_rec.append(animo_go_metric['rec_at_'+str(k)])

    fmax_val = fmax.f_max ( truth_np[: , index], prediction_np[: , index], threshold=np.arange(0,1,.02) )
    print ('fmax value {}'.format ( fmax_val ) ) 
    print ('precision/recall at K')
    print (track_prec)
    print (track_rec)



if len(sys.argv)<1: ## run script 
	print("Usage: \n")
	sys.exit(1)
else:
	submitJobs ( sys.argv[1] , sys.argv[2] , sys.argv[3] , sys.argv[4], int(sys.argv[5]) ) 
	




## when test against training using only GO at least 10 occ. 
# [MACRO] accuracy, precision, recall, f-measure, AUC
# 0.0304, 0.0474, 0.0374, 0.0418, 0.7235
# [MICRO] accuracy, precision, recall, f-measure, AUC
# 0.2712, 0.5086, 0.3674, 0.4266, 0.9032
# rec_at_5: 0.0705
# prec_at_5: 0.7019
# hamming loss 0.0027
# fmax value 0.4874582953329843




# [MACRO] accuracy, precision, recall, f-measure, AUC
# 0.0305, 0.0485, 0.0373, 0.0422, 0.7200             
# [MICRO] accuracy, precision, recall, f-measure, AUC
# 0.2720, 0.5125, 0.3670, 0.4277, 0.9017             
# rec_at_5: 0.0706                                   
# prec_at_5: 0.7025                                  
# hamming loss 0.002717215654613312                  
                                                   
                                                   
# [MACRO] accuracy, precision, recall, f-measure, AUC
# 0.0305, 0.0485, 0.0373, 0.0422, 0.7200             
# [MICRO] accuracy, precision, recall, f-measure, AUC
# 0.2720, 0.5125, 0.3670, 0.4277, 0.9017             
# rec_at_10: 0.1319                                  
# prec_at_10: 0.6648                                 
# hamming loss 0.002717215654613312                  
                                                   
                                                   
# [MACRO] accuracy, precision, recall, f-measure, AUC
# 0.0305, 0.0485, 0.0373, 0.0422, 0.7200             
# [MICRO] accuracy, precision, recall, f-measure, AUC
# 0.2720, 0.5125, 0.3670, 0.4277, 0.9017             
# rec_at_15: 0.1886                                  
# prec_at_15: 0.6345                                 
# hamming loss 0.002717215654613312                  
                                                   
                                                   
# [MACRO] accuracy, precision, recall, f-measure, AUC
# 0.0305, 0.0485, 0.0373, 0.0422, 0.7200             
# [MICRO] accuracy, precision, recall, f-measure, AUC
# 0.2720, 0.5125, 0.3670, 0.4277, 0.9017             
# rec_at_20: 0.2394                                  
# prec_at_20: 0.6140                                 
# hamming loss 0.002717215654613312                  

# fmax value 0.4825572967759702


