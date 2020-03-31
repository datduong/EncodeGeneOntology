
import pickle,os,sys,re
import pandas as pd
import numpy as np
from tqdm import tqdm


#### do not have mf/bp/cc . they jointly train all labels together.

def GetProtEmb (row):
  newstr = ";".join(str(r) for r in row) ## return string now, ... later we convert to tensor ## 256 prot-prot emb dim
  return newstr

os.chdir("/local/auppunda/auppunda/deepgoplus/deepgoplus.bio2vec.net/data/data-cafa")

data_types = ['train','test']
ontology_type = ['']


longest_len = 0.0

for ontology in ontology_type:

  go_name_array = pd.read_pickle("terms.pkl") #### terms used in prediction
  go_name_array = sorted( list ( go_name_array['terms'] ) ) ## do not sort, keep in original ordering ??? let's sort...
  print ('\nnum of terms to be predicted {}'.format(len(go_name_array)))
  fout = open ( '/local/datdb/deepgoplus/deepgoplus.bio2vec.net/data-cafa/data/deepgoplus.cafa3.label'+ontology+'.tsv', "w" )
  fout.write ("\n".join(g for g in go_name_array))
  fout.close()

  for data_type in data_types:

    print ('\n\n'+data_type+'-'+ontology)

    ## COMMENT original file downloaded from DeepGO does not show all GO terms being propagated (i.e. added with parents)
    ## COMMENT file contains 1-hot which already has all labels to be tested (found in mf.pkl)
    ## example
    # proteins                                           T100900000026
    # sequences      MAESFKELDPDSSMGKALEMTCAIQNQLARILAEFEMTLERDVLQP...
    # annotations    {GO:0065007, GO:0016043, GO:0051336, GO:001678...

    df = pd.read_pickle(data_type+'_data.pkl')
    df = df.reset_index(drop=True)
    print ('num obs x num row {}'.format(df.shape))
    print (df.columns)
    df.columns = ['Entry','Sequence','Gene ontology IDs']
    df = df[['Entry','Gene ontology IDs','Sequence']] ##!! swap the columns

    #### need to format GO terms into the xyz;xyz;xyz format
    label_array = []
    seq_len = []
    for index,row in df.iterrows():
      this_go = sorted ( list ( row['Gene ontology IDs'] ) )
      label = ";".join ( g for g in this_go if g in go_name_array ) ##!! only labels needed
      label_array.append(label)
      seq_len.append( len( row['Sequence'] ) )
      # if len( row['Sequence'] ) > longest_len:
      #   longest_len = len( row['Sequence'] )


    df['Gene ontology IDs'] = label_array ## assign go term found
    df['SequenceLen'] = seq_len

    #### remove long length
    df = df[ df['SequenceLen'] < 2001 ]
    df = df.reset_index(drop=True)

    ## COMMENT **** write to text ... need format " Entry Gene ontology IDs Sequence "
    df.to_csv('/local/datdb/deepgoplus/deepgoplus.bio2vec.net/data-cafa/data/deepgoplus.cafa3.'+data_type+'.tsv',sep="\t",index=False)

####
print ('\n\nlongest_len')
print (longest_len)



