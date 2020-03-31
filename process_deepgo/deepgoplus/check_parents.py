


from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata, string, re, sys, os, pickle, pickle, gzip
from tqdm import tqdm
import numpy as np
import pandas as pd

from copy import deepcopy

import networkx
import obonet


## what if we remove parents ?
## do we need to have complete path ?

os.chdir("/local/auppunda/auppunda/deepgoplus/deepgoplus.bio2vec.net/data/data-cafa")

terms = pd.read_pickle('terms.pkl')
terms ## [5220 rows x 1 columns]

#### train/test contains 29,000 terms. but they filter only >50 in their model
this_onto = []
for data_type in ['train']:  ##!! check that we have same number of GO in @terms as in the train.pkl
  df = pd.read_pickle(data_type+'_data.pkl')
  for index, this_set in tqdm ( enumerate (list (df['annotations']) ) ) :
    this_onto = this_onto + list (this_set) #.split(';')
    if index % 1000 == 1:
      this_onto = list(set(this_onto))
#
this_onto = sorted (list(set(this_onto)))


#### all their data are already expanded to contain parent terms

graph = obonet.read_obo('go.obo') # https://github.com/dhimmel/obonet

for data_type in ['train','test']:
  print ('\n'+data_type+'\n')
  df = pd.read_pickle(data_type+'_data.pkl')
  for i in [0,1,100,1000]:
    print ('\ncheck row '+str(i))
    node_list = df.iloc[0]['annotations']
    parent_at_node = []
    for node_name in node_list :
      for child, parent, key in graph.out_edges(node_name, keys=True):
        # @key tells "is_a" or "part_of" ...
        if key == 'is_a':
          parent_at_node.append(parent)
    ##
    ## check what we have must includes these parents
    not_in_list = set(parent_at_node) - set(node_list)
    print (not_in_list) ## should be empty to show that @node_list has all the parents of each child term.


