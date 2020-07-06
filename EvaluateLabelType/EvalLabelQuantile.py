

import re,os,sys,pickle
import numpy as np
import pandas as pd

sys.path.append("/u/scratch/d/datduong/EncodeGeneOntology/ProtSeq2GO")
import evaluation_metric

roots = ['GO0008150','GO0003674','GO0005575']
roots_with_colon = ['GO:0008150','GO:0003674','GO:0005575']

def GetCountDict (filename):
  count = {}
  name = []
  fin = open (filename,'r')
  for line in fin:
    line = line.strip().split()
    line[0] = re.sub("GO:","GO",line[0]) ## keep GOxyz
    count[line[0]] = int (line[1])
    name.append(line[0])
  fin.close()
  return count, name

def GetNumObsPerQuantile (count_dict,q=[.25,.75]):
  count = [count_dict[k] for k in count_dict]
  quant = np.quantile(count,q)
  print ( '\nquantiles {}\n'.format(quant) )
  return quant

def GetIndexOfLabelInQuantRange (label_to_test,count_dict):
  quantile_range = GetNumObsPerQuantile(count_dict) # ! 25 and 75 for now.
  quantile_index = {'25':[], '25-75':[], '75':[] }
  for index, label in enumerate (label_to_test): # ! keep same ordering as in input file. this file should already be sorted.
    if count_dict[label] < quantile_range[0] : # less 25%
      quantile_index['25'].append(index)
    elif count_dict[label] > quantile_range[1] : # over 75%
      quantile_index['75'].append(index)
    else:
      quantile_index['25-75'].append(index)
  #
  for key,value in quantile_index.items():
    quantile_index[key] = np.array (value) ## convert to np for indexing
  return quantile_index

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

def submitJobs (label_path, onto, load_path):

  IC_dict = None # pickle.load(open("/u/scratch/d/datduong/deepgoplus/data-cafa/ICsValueTable.pickle","rb"))
  # for onto in ['cc','mf','bp']:

  label_count , label_name = GetCountDict(label_path)

  prediction_dict = pickle.load(open(load_path,"rb"))

  print ('\nmodel {} type {}'.format(load_path, onto ))
  print ('\nsize {}\n'.format(prediction_dict['prediction'].shape))

  print ('\nwhole')
  #! remove roots.
  where_not_roots = [ i for i,name in enumerate(label_name) if name not in roots ]
  #! the IC require GO:xyz format
  label_name = [ re.sub('GO','GO:',lab) for lab in label_name ]
  label_name = np.array(label_name)
  label_name = label_name [ where_not_roots ]

  for key in prediction_dict:
    prediction_dict[key] = prediction_dict[key][: , where_not_roots] #? filter col

  evaluation_metric.print_metrics( eval(prediction_dict, None, IC_dict, label_name ) )

  # print ('\n\neval by our code version\n')
  # evaluation_metric.print_metrics( eval(prediction_dict ) )

  #! get accuracy based on quantile count
  for r in roots_with_colon:
    if r in label_count:
      print ('remove root from label count')
      del label_count[r]

  label_name = [ re.sub('GO:','GO',lab) for lab in label_name ] ## put back naming convention GO:xyz
  quantile_index = GetIndexOfLabelInQuantRange(label_name,label_count)
  label_name = [ re.sub('GO','GO:',lab) for lab in label_name ] #!! SO STUPID

  for quant in ['25','25-75','75']:
    print('\nq {}'.format(quant))
    evaluation_metric.print_metrics( eval(prediction_dict, quantile_index[quant], IC_dict, label_name) )


if len(sys.argv)<1: #### run script
	print("Usage: \n")
	sys.exit(1)
else:
	submitJobs ( sys.argv[1], sys.argv[2], sys.argv[3] )


