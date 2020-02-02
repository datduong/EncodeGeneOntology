

import re,os,sys,pickle
import numpy as np
import pandas as pd
main_dir = '/local/datdb/deepgo/data/train/fold_1'
# main_dir = '/local/datdb/deepgo/dataExpandGoSet16Jan2020/train/'
os.chdir( main_dir )

sys.path.append("/local/datdb/GOmultitask")
import ProtSeq2GO.ProtSeqLoader as ProtSeqLoader

def GetNumObsPerQuantile (count_dict,q):
  count = [count_dict[k] for k in count_dict]
  quant = np.quantile(count,q) ## probably 2 of these are enough
  return quant

def GetIndexOfLabel (label_to_test,category):
  quantile_index = {}
  # for category in ['bp','mf','cc']:
  GO_counter = pickle.load(open("CountGoInTrain-"+category+".pickle","rb"))
  print ('\n\ncategory {}, count {}'.format(category,len(label_to_test)))
  quant25, quant75 = ProtSeqLoader.GetNumObsPerQuantile(GO_counter)
  print ('value 25 and 75 quantiles {} {}'.format(quant25, quant75) )
  print ('value 50 {}'.format(GetNumObsPerQuantile(GO_counter,[0.5])))
  betweenQ25Q75 = ProtSeqLoader.IndexInRangeQuantileXY(label_to_test,GO_counter,quant25,quant75)
  quant25 = ProtSeqLoader.IndexBelowQuantileX(label_to_test,GO_counter,quant25)
  quant75 = ProtSeqLoader.IndexOverQuantileX(label_to_test,GO_counter,quant75)
  print ('num label in 25 and 75 quantiles {} {}'.format(len(quant25), len(quant75)))
  quantile_index['25']=quant25
  quantile_index['75']=quant25
  quantile_index['25-75']=betweenQ25Q75
  return quantile_index


#### eval accuracy of labels ...

sys.path.append("/local/datdb/GOmultitask")
import ProtSeq2GO.evaluation_metric as evaluation_metric

def eval (prediction_dict,sub_array=None):
  prediction = prediction_dict['prediction']
  true_label = prediction_dict['truth']
  if sub_array is not None:
    prediction = prediction [ : , sub_array ] ## obs x label
    true_label = true_label [ : , sub_array ]
  #
  result = evaluation_metric.all_metrics ( np.round(prediction) , true_label, yhat_raw=prediction, k=[5,10,15,20,25,30,35,40])
  return result

def submitJobs (method):

  os.chdir('/local/datdb/deepgo/data/train/fold_1/')

  # method = 'BiLSTM.768'

  for onto in ['cc','mf','bp']:

    label_original = pd.read_csv('/local/datdb/deepgo/data/train/deepgo.'+onto+'.csv',sep="\t",header=None)
    label_original = sorted(list(label_original[0])) ##!! do not sort here.

    quantile_index = GetIndexOfLabel(label_original,onto)

    load_path = method+'/'+onto+'b32lr0.001'
    prediction_dict = pickle.load(open(load_path+"/prediction_testset.pickle","rb"))

    print ('\nmodel {} type {}'.format(method, onto ))
    print ('\nsize {}\n'.format(prediction_dict['prediction'].shape))

    print ('\nwhole')
    evaluation_metric.print_metrics( eval(prediction_dict) )

    for quant in ['25','25-75','75']:
      print('\n {}'.format(quant))
      evaluation_metric.print_metrics( eval(prediction_dict, quantile_index[quant]) )



if len(sys.argv)<1: #### run script
	print("Usage: \n")
	sys.exit(1)
else:
	submitJobs ( sys.argv[1] )


