from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string, re, sys, os, pickle, gzip
from tqdm import tqdm
import numpy as np
import pandas as pd
from collections import namedtuple
from tempfile import TemporaryDirectory

import compare_set


root = ['GO:0008372','GO:0005575','GO:0008150','GO:0000004', 'GO:0007582', 'GO:0044699', 'GO:0003674',' GO:0005554']


# main_dir = '/u/scratch/d/datduong/goAndGeneAnnotationDec2018'
main_dir = '/u/scratch/d/datduong/goAndGeneAnnotationMar2017'
os.chdir(main_dir)

## read all GO def
## should we use the old 2017 era go definitions ? 
GOdef = compare_set.def2dict(name="go_def_in_obo.tsv")
GO_name_array = list(GOdef.keys())

""" before we remove tags 
["\tIEA\t", "\tNAS\t", "\tNA\t", "\tNR\t"]
But there were changes, some tags don't exists anymore. We will use original file in the format like this goa_fly.gafNotEI 
"""

# ### !! create species annot
# os.chdir('/u/scratch/d/datduong/goAndGeneAnnotationMar2017/gafData2017')
# for spec in ['human'] : # ['fb','wb','mgi','sgd']: # 'fb','wb','mgi', 'goa_human','fb','wb','mgi'
#   df = pd.read_csv( "goa_"+spec+".gaf", header=None, sep="\t", comment="!", dtype='str' )
#   # df = pd.read_csv( "gene_association."+spec+"", header=None, sep="\t", comment="!", dtype='str' )
#   print ('table size {}'.format(df.shape))
#   ### !!!! 
#   """ before we remove tags 
#   ["\tIEA\t", "\tNAS\t", "\tNA\t", "\tNR\t"]
#   But there were changes, some tags don't exists anymore. We will use original file in the format like this goa_fly.gaf. We also have goa_fly.gafNotEI but it seems the spacing is off somehow. so that tab space is not detected right.
#   """
#   df = df [ ~df[6].isin(['IEA','NAS','NA','NR']) ]
#   df = df [ ~df[3].isin(['NOT']) ]
#   df = df [ df[4].isin(GO_name_array) ] ## only valid def
#   df = df [ [2,4] ] ## get these col only
#   df = df.dropna() ## empty string name ?? for fly base ??
#   df = df.sort_values(by=[2])
#   df = df.drop_duplicates ()
#   df.columns = ['uniprot','go']
#   print ('table size after remove IEA {}'.format(df.shape))
#   df.to_csv( spec+'_not_IEA.tsv',sep="\t",index=False)


def submitJobs (gaf1,gaf2,saveDf,savePickle,genePairList,start,gapSize):

  genePairList = pd.read_csv(genePairList,sep=" ",dtype=str) # header=None

  if start > genePairList.shape[0]:
    exit() 
    
  end = start + gapSize
  if end > genePairList.shape[0]:
    end = genePairList.shape[0]

  genePairList = genePairList[start:end] ## subset

  # gaf1 = 'goa_human_not_IEA.tsv'
  GeneGOdb1 = compare_set.gaf2dict(gaf1)

  # gaf2 = 'mgi_not_IEA.tsv'
  GeneGOdb2 = compare_set.gaf2dict(gaf2)

  # dfPair = '/u/scratch/d/datduong/geneOrtholog/HumanMouseScore/HumanMouseOrtholog2test.txt'
  geneDict = compare_set.GenePairDict (genePairList)

  geneDict.make_pair(GeneGOdb1,GeneGOdb2)

  geneDict.write_go_pairs(GOdef,saveDf) # '/u/scratch/d/datduong/geneOrtholog/HumanMouseScore/HumanMouseOrtholog2testDef.txt'

  print ('num GO pairs {}'.format(len(geneDict.LargeGOpair)))
  print ('num gene pairs {}'.format(len(geneDict.genePair)))

  pickle.dump(geneDict,open(savePickle,'wb')) # '/u/scratch/d/datduong/geneOrtholog/HumanMouseScore/geneDict2test.pickle'

  # geneDict.genePair['CLP1,Pigv'][0].index


if len(sys.argv)<1: ## run script
	print("Usage: \n")
	sys.exit(1)
else:
	submitJobs ( sys.argv[1] , sys.argv[2] , sys.argv[3] , sys.argv[4], sys.argv[5], int(sys.argv[6]), int(sys.argv[7]) )

