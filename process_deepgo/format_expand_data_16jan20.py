



import pickle,os,sys,re
import pandas as pd
import numpy as np
from copy import deepcopy

# swissprot = pd.read_pickle('swissprot_exp.pkl') #
# list(swissprot['accessions']).index('Q39218')

def GetProtEmb (row):
  newstr = ";".join(str(r) for r in row) ## return string now, ... later we convert to tensor ## 256 prot-prot emb dim
  return newstr

# BIOLOGICAL_PROCESS = 'GO:0008150'
# MOLECULAR_FUNCTION = 'GO:0003674'
# CELLULAR_COMPONENT = 'GO:0005575'
root_nodes = ['GO:0008150','GO:0003674','GO:0005575']

os.chdir("/local/datdb/deepgo/dataExpandGoSet16Jan2020")

# data_type = 'test'
# ontology = 'mf'

for data_type in ['train','test']:
  for ontology in ['mf','bp','cc']:

    df = pd.read_pickle(data_type+'-'+ontology+'.pkl') # read in the pickle
    df = df.drop(columns=['labels']) ## don't need their 1-hot

    prot = list(df['accessions'])
    prot_to_remove = []
    label_array = []
    emb_array = []

    #### get the labels to be predicted
    label_array_from_data = list(df['gos'])
    go_name_counter = {}
    for counter, this_label in enumerate ( label_array_from_data ):
      for l in this_label:
        if l not in go_name_counter:
          go_name_counter[l] = 1
        else:
          go_name_counter[l] = 1 + go_name_counter[l] ## keep count
        #

    go_name_array = list(go_name_counter.keys())
    go_name_array = sorted(list(go_name_array)) ## sort by alphabet, who cares about the 1-hot, we will make them again anyway
    print ('\ncount term {}'.format(len(go_name_array)))
    # CountGoInTest-mf
    fout = open ('CountGoIn-'+data_type+'-'+ontology+'.tsv', 'w')
    fout.write('GO\tcount\n')
    for key in go_name_array:
      fout.write(key + "\t" + str(go_name_counter[key]) + "\n")

    #
    fout.close()

    #### load label pickle, double check with train file
    label_from_pickle = pickle.load(open(ontology+'expanded.pkl','rb'))
    label_from_pickle = sorted ( list (label_from_pickle['functions']) ) ##!!##!!##!!##!!##!!##!!
    in_file_but_not_complete_list = set (go_name_array) - set (label_from_pickle)
    # set (label_from_pickle) - set (go_name_array)
    if len(in_file_but_not_complete_list) > 0:
      print ('check label')
      exit()

    fout = open ( 'deepgo.'+ontology+'.csv', "w" )
    fout.write ("\n".join(g for g in label_from_pickle))
    fout.close()

    go_name_array = deepcopy(label_from_pickle) ####


    #### create an array of label for each protein

    for counter, this_label in enumerate ( label_array_from_data ):
      this_label = [ term for term in this_label if term not in root_nodes ] ## must remove root, otherwise the root will be all proteins
      this_label = [ term for term in this_label if term in go_name_array ] ## make sure we get only GO in mf or bp or cc
      if len(this_label) == 0:
        label_array.append ([""]) ## nothing
        prot_to_remove.append(prot[counter])
      else:
        label = ";".join ( g for g in sorted(this_label) )
        label_array.append ( label )
      ## get prot emb in string format
      emb_array.append ( GetProtEmb(df.iloc[counter]['embeddings']) )


    #### write output
    df['Gene ontology IDs'] = label_array ## assign go term found
    df['Prot Emb'] = emb_array

    print ( df.iloc[0] )
    print ('remove these prot because no GO assign to them')
    print (prot_to_remove)

    df = df[~df['accessions'].isin(prot_to_remove)] ## remove prot without any term
    print ('final dim {}'.format(df.shape))

    df = df[['accessions','Gene ontology IDs','sequences','Prot Emb']]
    df.columns = ['Entry','Gene ontology IDs','Sequence','Prot Emb']

    df.to_csv(data_type+'-'+ontology+'-16Jan20.tsv',sep="\t",index=False)


