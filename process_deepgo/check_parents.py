


from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata, string, re, sys, os, pickle, pickle, gzip

import numpy as np
import pandas as pd 

from copy import deepcopy

import networkx
import obonet


## what if we remove parents ?
## do we need to have complete path ? 

work_dir = '/u/flashscratch/d/datduong/deepgo/data/'
os.chdir (work_dir)

# Read the taxrank ontology
graph = obonet.read_obo('go.obo') # https://github.com/dhimmel/obonet

node_list = ['GO:0008270', 'GO:0043167', 'GO:0046914', 'GO:0043169', 'GO:0046872', 'GO:0005488', 'GO:0016413', 'GO:0016407', 'GO:0016412', 'GO:0003824', 'GO:0008374', 'GO:0016747', 'GO:0016740', 'GO:0016746']

for node_name in node_list : 
  parent_at_node = []
  for child, parent, key in graph.out_edges(node_name, keys=True):
    # @key tells "is_a" or "part_of" ...
    if key == 'is_a': 
      parent_at_node.append(parent)
  ##
  ## check what we have must includes these parents 
  not_in_list = set(parent_at_node) - set(node_list) 
  if len( not_in_list ) > 0 or ('GO:0016413' in parent_at_node): 
    print (node_name)
    print (not_in_list)



check = ['GO:0008270', 'GO:0005739', 'GO:0019344', 'GO:0006535', 'GO:0009001', 'GO:0008652', 'GO:0046686']
for n in check: 
  all_anc = []
  node_list = [n]
  while True: 
    for node_name in node_list : 
      parent_at_node = []
      for child, parent, key in graph.out_edges(node_name, keys=True):
        # @key tells "is_a" or "part_of" ...
        if key == 'is_a': 
          parent_at_node.append(parent)
    ## 
    if len(parent_at_node)==0: 
      break 
    if  ('GO:0016413' in parent_at_node):
      print (node_name)
      print (parent_at_node)
    all_anc = all_anc + parent_at_node
    node_list = deepcopy(parent_at_node) ## go up 1 level 
    


# ###
# Q39218  GO:0008270;GO:0046914;GO:0046872;GO:0043169;GO:0043167;GO:0005488;GO:0008374;GO:0016407;GO:0016747;GO:0016746;GO:0016740;GO:0003824
# >>> sorted ( df.iloc[0]['gos'] )
# ['GO:0008270', 'GO:0043167', 'GO:0046914', 'GO:0043169', 'GO:0046872', 'GO:0005488', 'GO:0016413', 'GO:0016407', 'GO:0016412', 'GO:0003824', 'GO:0008374', 'GO:0016747', 'GO:0016740', 'GO:0016746']

# {'GO:0043169', 'GO:0005488', 'GO:0003674', 'GO:0046914', 'GO:0046872', 'GO:0043167'}

# {'GO:0044424', 'GO:0044464', 'GO:0043227', 'GO:0005622', 'GO:0043226', 'GO:0005575', 'GO:0005623', 'GO:0044444', 'GO:0043229', 'GO:0043231', 'GO:0005737'}

# GO:0009001
# ['GO:0016412', 'GO:0016413']

# ['GO:0008270|IDA', 'GO:0005739|IDA', 'GO:0019344|IMP', 'GO:0006535|IEA', 'GO:0009001|IDA', 'GO:0008652|IEA', 'GO:0046686|IEP']
