
import pickle,os,sys,re
import pandas as pd
import numpy as np
from tqdm import tqdm 

#### example to read in pickle, notice we use @labels field and not @go field in pandas.
# df = pd.read_pickle('/local/datdb/deepgo/data/train/train-mf.pkl') #
# label_array_from_data = list(df['labels']) ## this is 1-hot
# not_found = 0
# for l in label_array_from_data: 
#   if sum(l) == 0: 
#     not_found = not_found + 1
# #
# # check size of prot with labels
# print (not_found)
# print (df.shape[0]-not_found) ### 25198
# # 6294 test
# # 20158 train 
# # 5040 dev 


## load save pickle
## combine them ? probably not need to.

def GetProtEmb (row):
  newstr = ";".join(str(r) for r in row) ## return string now, ... later we convert to tensor ## 256 prot-prot emb dim
  return newstr

## COMMENT we used our own expansion method for filter >50 >10 >10, so we don't have a pickle file.

os.chdir("/u/scratch/d/datduong/deepgoCheck/data/train")

data_types = ['train','test']
ontology_type = ['bp'] # 'mf','cc',

for ontology in ontology_type: ## need to create mf.pkl and so forth
  this_onto = []
  for data_type in data_types: 
    df = pd.read_pickle(data_type+'-'+ontology+".pkl")
    # df = pd.read_csv(data_type+'-'+ontology+".tsv",sep="\t")
    for index, this_set in tqdm ( enumerate (list (df['gos']) ) ) : 
      this_onto = this_onto + this_set #.split(';')
      # if 'GO:2001272' in this_set: # GO:2001272
      #   print (index)
      if index % 1000 == 1: 
        this_onto = list(set(this_onto))
  #
  this_onto = sorted (list(set(this_onto)))
  df = pd.DataFrame({'functions':this_onto})
  # pd.to_pickle(df,ontology+".pkl")


for ontology in ontology_type:

  go_name_array = pd.read_pickle(ontology+".pkl") ## terms used in prediction
  go_name_array = np.array ( list ( go_name_array['functions'] ) ) ## do not sort, keep in original ordering
  fout = open ( 'deepgo.'+ontology+'.csv', "w" )
  fout.write ("\n".join(g for g in go_name_array))
  fout.close()

  for data_type in data_types:

    print ('\n\n'+data_type+'-'+ontology)

    ## COMMENT original file downloaded from DeepGO does not show all GO terms being propagated (i.e. added with parents)
    ## COMMENT file contains 1-hot which already has all labels to be tested (found in mf.pkl)
    ## example
    # df.iloc[0]
    # accessions                                               Q9VSY6
    # gos           [GO:0007411, GO:0030431, GO:0045433, GO:004814...
    # labels        [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...
    # ngrams        [4306, 6116, 2318, 6350, 6996, 3910, 6181, 361...
    # proteins                                             SERB_DROME
    # sequences     MSGSVLSLARPAAATNGHNLLTKQLNCNGNGTTGGAAKTTVASAIT...
    # orgs                                                       7227
    # embeddings    [-0.484184, -0.022484, -0.204611, 0.072551, -1...
    # Name: 41301, dtype: object
    # >>> np.sum(df.iloc[0]['labels'])
    # 17
    # >>> len(df.iloc[0]['gos'])
    # 5
    # >>> len(df.iloc[0]['labels'])
    # 932

    df = pd.read_pickle(data_type+'-'+ontology+".pkl") 
    df = df.reset_index(drop=True)
    print ('num obs x num row {}'.format(df.shape))
    print (df.columns)
    # df.columns
    # Index(['accessions', 'gos', 'labels', 'ngrams', 'proteins', 'sequences',
    #        'orgs', 'embeddings'],

    # df['sequences'][0]
    # df['gos'][0]

    # where1 = np.where ( df.iloc[8]['labels'] == 1 ) [0]
    # # ['GO:0008426']
    # go_name_array[where1]

    ## COMMENT **** write to text ... need format " Entry Gene ontology IDs Sequence "

    label_array_1hot = list(df['labels']) ## array of array of 1-hot. this is original label set AND parents added
    prot = list(df['accessions'])
    prot_to_remove = []
    label_array = []
    emb_array = []

    for counter, this_label in enumerate ( label_array_1hot ):
      where1 = np.where ( this_label == 1 ) [0]
      if len(where1) == 0:
        label_array.append ([""]) ## nothing
        prot_to_remove.append(prot[counter])
      else:
        label = ";".join ( g for g in go_name_array[where1] )
        label_array.append ( label )

      ## get prot emb in string format
      emb_array.append ( GetProtEmb(df.iloc[counter]['embeddings']) )


    df['Gene ontology IDs'] = label_array ## assign go term found
    df['Prot Emb'] = emb_array

    print ('remove these prot because no GO assign to them')
    print (prot_to_remove)

    df = df[~df['accessions'].isin(prot_to_remove)] ## remove prot without any term
    print ('final dim {}'.format(df.shape))

    df = df[['accessions','Gene ontology IDs','sequences','Prot Emb']]
    df.columns = ['Entry','Gene ontology IDs','Sequence','Prot Emb']

    df.to_csv(data_type+'-'+ontology+'.tsv',sep="\t",index=False)



