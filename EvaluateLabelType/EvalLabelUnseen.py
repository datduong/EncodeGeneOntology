
import sys,re,os,pickle
import pandas as pd
import numpy as np

# sys.path.append("/local/datdb/GOmultitask")
# import ProtSeq2GO.evaluation_metric as evaluation_metric

sys.path.append("/local/datdb/EncodeGeneOntology/ProtSeq2GO")
import evaluation_metric

#### eval accuracy of unseen labels ...

# def eval (prediction_dict,sub_array=None):
#   prediction = prediction_dict['prediction']
#   true_label = prediction_dict['truth']
#   if sub_array is not None:
#     prediction = prediction [ : , sub_array ] ## obs x label
#     true_label = true_label [ : , sub_array ]
#   #
#   # k=[5,10,15,20,25,30,35,40]
#   result = evaluation_metric.all_metrics ( np.round(prediction) , true_label, yhat_raw=prediction, k=10 )
#   return result


def eval (prediction_dict,sub_array=None,IC_dict=None,label_name=None): ## ! eval accuracy of labels ...

  prediction = prediction_dict['prediction']
  key_name = 'truth'
  if key_name not in prediction_dict: # ! different save pickle has different names
    key_name = 'true_label'
  #
  true_label = prediction_dict[key_name] # true_label, truth
  if sub_array is not None:
    prediction = prediction [ : , sub_array ] ## obs x label
    true_label = true_label [ : , sub_array ]
  #

  #! remove 1 single protein MF that had just root as label ??
  #! remove any protein that has no true label?? this will change recall@k
  has_true = np.where ( true_label.sum(1) > 0 ) [0] ## add up row, remove proteins with no true label
  true_label = true_label [ has_true , ]
  prediction = prediction [ has_true , ]

  print ('new filter size {}'.format(prediction.shape))

  result = evaluation_metric.all_metrics ( np.round(prediction) , true_label, yhat_raw=prediction, k=np.arange(10,110,10).tolist(), IC_dict=IC_dict, label_names=label_name )
  return result


def submitJobs (where,method,run_param_name):

  # os.chdir('/local/datdb/deepgo/data/train/fold_1/') ###
  os.chdir(where)

  for onto in ['cc','mf','bp']:

    label_original = pd.read_csv('/local/datdb/deepgo/data/train/deepgo.'+onto+'.csv',sep="\t",header=None)
    label_original = set(list(label_original[0])) ##!! do not sort here, so that we can use set-subtraction

    label_large = pd.read_csv('/local/datdb/deepgo/dataExpandGoSet16Jan2020/train/deepgo.'+onto+'.csv',sep="\t",header=None)
    label_large = set(list(label_large[0]))

    label_unseen = sorted ( list ( label_large - label_original ) )
    label_large = sorted(label_large) ##!!##!! by default we sort label for the model

    ##!!##!! get index of each label. {labelX:1 , labelY:4}
    label_lookup = {value:counter for counter,value in enumerate(label_large)}
    label_unseen_pos = np.array ( [label_lookup[v] for v in label_lookup if v in label_unseen ] )

    label_original = sorted(list(label_original))
    label_seen_pos = np.array ( [label_lookup[v] for v in label_lookup if v in label_original ] ) ## doesn't really need to be sorted, because we extract by index anyway @label_lookup

    load_path = method+'/'+onto+run_param_name ##'b32lr0.0005'
    print ('load in save {}'.format(load_path))
    try: 
      prediction_dict = pickle.load(open(load_path+"/test-"+onto+"-same-origin.pickle","rb")) ##!!##!!
      # prediction_dict = pickle.load(open(load_path+"/prediction_testset.pickle","rb")) ##!!##!!
    except: 
      continue

    print ('\nmodel {} type {}'.format(method, onto ))
    print ('\nsize {}\n'.format(prediction_dict['prediction'].shape))

    # print ('\nwhole ' + onto)
    # evaluation_metric.print_metrics( eval(prediction_dict) )

    print('\noriginal ' + onto)
    evaluation_metric.print_metrics( eval(prediction_dict, label_seen_pos) )

    print ('\nadded ' + onto)
    evaluation_metric.print_metrics( eval(prediction_dict, label_unseen_pos) )


if len(sys.argv)<1: #### run script
	print("Usage: \n")
	sys.exit(1)
else:
	submitJobs ( sys.argv[1] , sys.argv[2] , sys.argv[3] )


