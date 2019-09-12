


## count how many times a go term occur
import re,os,sys,pickle
import numpy as np 
import pandas as pd

# main_dir = '/u/scratch/d/datduong/deepgo/data/train/'
main_dir = '/u/scratch/d/datduong/deepgo/dataExpandGoSet/train/'
os.chdir( main_dir )


# "train-"
all_counter = {}
for data_type in ["train-", "test-"] :
  for category in ['cc','bp','mf']: 
    go_counter = {} 
    df = pd.read_csv(data_type+category+".tsv",sep="\t")
    for counter, row in df.iterrows(): # Entry Gene ontology IDs Sequence  Prot Emb
      go = row['Gene ontology IDs'].strip().split(";")
      for g in go: 
        if g in go_counter: 
          go_counter[g] = go_counter[g] + 1
        else: 
          go_counter[g] = 1
    ## 
    ## for each category, we save the output
    print (data_type+category)
    all_counter[data_type+category] = go_counter
    # write go counter to file to plot 
    ## name 
    name_out = re.sub(r"^t","T",data_type)
    name_out = re.sub(r"-","",name_out)
    fout = open("CountGoIn"+name_out+"-"+category+".tsv",'w')
    fout.write("GO\tcount\n")
    for k,v in go_counter.items(): 
      fout.write(k+"\t"+str(v)+"\n")
    fout.close() 
    pickle.dump(go_counter, open("CountGoIn"+name_out+"-"+category+".pickle","wb"))


## 

# set_test = set(all_counter['train-cc'].keys())
# set_train = set(all_counter['test-cc'].keys())
# set_test-set_train

import re,os,sys,pickle
import numpy as np 
import pandas as pd
# main_dir = '/u/scratch/d/datduong/deepgo/data/train/fold_1'
main_dir = '/u/scratch/d/datduong/deepgo/dataExpandGoSet/train/'
os.chdir( main_dir )

sys.path.append("/u/scratch/d/datduong/GOmultitask")
import ProtSeq2GO.protSeqLoader as protSeqLoader

def GetCountQuantile (count_dict,q): 
  count = [count_dict[k] for k in count_dict] 
  quant = np.quantile(count,q) ## probably 2 of these are enough 
  return quant

for category in ['bp','mf','cc']:
  GO_counter = pickle.load(open("CountGoInTrain-"+category+".pickle","rb"))
  label_to_test = list ( GO_counter.keys() ) 
  print ('\n\ncategory {}, count {}'.format(category,len(label_to_test)))
  quant25, quant75 = protSeqLoader.GetCountQuantile(GO_counter)
  print ('value 25 and 75 quantiles {} {}'.format(quant25, quant75) ) 
  print ('value 50 {}'.format(GetCountQuantile(GO_counter,[0.5])))
  betweenQ25Q75 = protSeqLoader.IndexBetweenQ25Q75Quantile(label_to_test,GO_counter,quant25,quant75)
  quant25 = protSeqLoader.IndexLessThanQuantile(label_to_test,GO_counter,quant25)
  quant75 = protSeqLoader.IndexMoreThanQuantile(label_to_test,GO_counter,quant75)
  print ('counter 25 and 75 quantiles {} {}'.format(len(quant25), len(quant75)))




