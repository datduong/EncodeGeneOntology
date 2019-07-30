
import pickle,os,sys,re
import pandas as pd 
import numpy as np 

## load save pickle
## combine them ? probably not need to. 

def GetProtEmb (row):
  newstr = ";".join(str(r) for r in row) ## return string now, ... later we convert to tensor ## 256 prot-prot emb dim
  return newstr

os.chdir("/u/flashscratch/d/datduong/deepgo/data/train")

data_types = ['train','test']
ontology_type = ['cc','mf','bp']

for ontology in ontology_type: 

  go_name_array = pd.read_pickle(ontology+".pkl") ## terms used in prediction
  go_name_array = np.array ( list ( go_name_array['functions'] ) ) ## do not sort, keep in original ordering
  fout = open ( 'deepgo.'+ontology+'.csv', "w" )
  fout.write ("\n".join(g for g in go_name_array))
  fout.close() 

  for data_type in data_types: 
    
    print ('\n\n'+data_type+'-'+ontology)

    df = pd.read_pickle(data_type+'-'+ontology+".pkl") ## this file does not show all GO terms being propagated (i.e. added with parents) train-mf
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


    ## **** write to text ... need format " Entry Gene ontology IDs Sequence "

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



