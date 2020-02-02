


import pickle,os,sys,re
import pandas as pd
import numpy as np


# BIOLOGICAL_PROCESS = 'GO:0008150'
# MOLECULAR_FUNCTION = 'GO:0003674'
# CELLULAR_COMPONENT = 'GO:0005575'

root_nodes = ['GO:0008150','GO:0003674','GO:0005575']

# swissprot = pd.read_pickle('swissprot_exp.pkl') #
# list(swissprot['accessions']).index('Q39218')

## load save pickle
## combine them ? probably not need to.

def GetProtEmb (row):
  newstr = ";".join(str(r) for r in row) ## return string now, ... later we convert to tensor ## 256 prot-prot emb dim
  return newstr

out_dir = "/u/flashscratch/d/datduong/deepgo/dataExpandGoSet/train/"
in_dir = "/u/scratch/a/auppunda/deepgo2/data/train" ## COMMENT
os.chdir(out_dir)

data_types = ['train','test']
ontology_type = ['cc','mf','bp']

for ontology in ontology_type:

  go_name_array = pd.read_pickle(in_dir+"/"+ontology+"2.pkl") ## COMMENT terms used in prediction
  go_name_array = np.array ( list ( go_name_array['functions'] ) ) ## do not sort, keep in original ordering
  fout = open ( 'deepgo.'+ontology+'.csv', "w" )
  go_name_array = sorted(list(go_name_array)) ## sort by alphabet, who cares about the 1-hot, we will make them again anyway
  fout.write ("\n".join(g for g in go_name_array))
  fout.close()

  for data_type in data_types:

    print ('\n\n'+data_type+'-'+ontology)

    df = pd.read_pickle(in_dir+"/"+data_type+'-'+ontology+"-appended.pkl") ## SHOULD ALREADY has GO terms being propagated (i.e. added with parents) train-mf
    df = df.reset_index(drop=True)
    print ('num obs x num row {}'.format(df.shape))
    print (df.columns)

    ## **** write to text ... need format " Entry Gene ontology IDs Sequence "

    # label_array_1hot = list(df['labels']) ## array of array of 1-hot. this is original label set AND parents added
    ## COMMENT can't trust the 1-hot, it can be linked to the old model
    ## later, we make our own 1-hot anyway
    df = df.drop(columns=['labels'])

    prot = list(df['accessions'])
    prot_to_remove = []
    label_array = []
    emb_array = []

    label_array_from_data = list(df['gos'])
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


    df['Gene ontology IDs'] = label_array ## assign go term found
    df['Prot Emb'] = emb_array

    print ( df.iloc[0] )
    print ('remove these prot because no GO assign to them')
    print (prot_to_remove)

    df = df[~df['accessions'].isin(prot_to_remove)] ## remove prot without any term
    print ('final dim {}'.format(df.shape))

    df = df[['accessions','Gene ontology IDs','sequences','Prot Emb']]
    df.columns = ['Entry','Gene ontology IDs','Sequence','Prot Emb']

    df.to_csv(data_type+'-'+ontology+'.tsv',sep="\t",index=False)



