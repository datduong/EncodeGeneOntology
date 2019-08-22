

from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata, string, re, sys, os, pickle, pickle, gzip

from tqdm import tqdm

import numpy as np
import pandas as pd 

from copy import deepcopy

import networkx
import obonet

## using the COO format of sklearn 

## deepgo/data/train

# work_dir = '/u/flashscratch/d/datduong/goAndGeneAnnotationDec2018/'
work_dir = '/u/flashscratch/d/datduong/deepgo/data/'

os.chdir (work_dir)

# Read the taxrank ontology
graph = obonet.read_obo('go.obo') # https://github.com/dhimmel/obonet

len(graph) # Number of nodes

graph.number_of_edges() # Number of edges

networkx.is_directed_acyclic_graph(graph) # Check if the ontology is a DAG

# Mapping from term ID to name
id_to_name = {id_: data.get('name') for id_, data in graph.nodes(data=True) if 'OBSOLETE' not in data.get('def')} ## by default obsolete already removed
# id_to_name['GO:0000002']  


# Find all superterms of species. Note that networkx.descendants gets
# superterms, while networkx.ancestors returns subterms.
# networkx.descendants(graph, 'GO:0000002') ## return parents https://www.ebi.ac.uk/QuickGO/term/GO:0000002 GO:1904426

# the first list contains the index of the source nodes, 
# while the index of target nodes is specified in the second list.
# edge_index = torch.tensor([[0, 1, 2, 0, 3],
#                            [1, 0, 1, 3, 2]], dtype=torch.long)

short_name = {"mf":'molecular_function', "bp": 'biological_process', "cc":'cellular_component'}

for onto_type in ["mf","bp","cc"]: 

  all_term_in_onto_type = [ go for go in id_to_name if graph.node[go]['namespace'] == short_name[onto_type] ] ## only take out GO of this category 
  all_term_in_onto_type.sort() 
  look_up = {}
  for counter, val in enumerate (all_term_in_onto_type): 
    look_up[val] = counter

  fout = open("terms_in_"+onto_type+".csv","w") ## write for reference later
  fout.write ("\n".join(g for g in all_term_in_onto_type))
  fout.close() 

  df = pd.read_csv("deepgo."+onto_type+".csv",header=None,dtype=str)
  go_name_array_obo = sorted ( list ( df[0] ) ) 

  adjacency_matrix = np.zeros( (len(all_term_in_onto_type), len(all_term_in_onto_type)) ) # @adjacency_matrix : has row=node , col=1 for where child of node is at

  # https://github.com/dhimmel/obonet/blob/master/examples/go-obonet.ipynb
  for node_name in tqdm(go_name_array_obo): 

    source_index = look_up[node_name]
    adjacency_matrix[source_index , source_index] = 1 ## add self. 
    
    children = list ( networkx.ancestors(graph, node_name) ) ## notice some dumb notation, @ancestors is getting all the terms that @node_name is the ancestor of. 

    ##  **** NOTICE, CHILDREN NOT IN SAME ONTOLOGY ARE ALSO RETURNED. 
    where = [ look_up[ch] for ch in children if graph.node[ch]['namespace'] == short_name[onto_type] ]
    if len (where) > 0: 
      children_index = np.array ( where)
      adjacency_matrix[source_index , children_index] = 1 ## get all descendent, row=node, col=children
    # else: 
    #   print (node_name)
    #   exit() 


  pickle.dump ( adjacency_matrix , gzip.open("adjacency_matrix_"+onto_type+".gzip.pickle","wb"), protocol=4 )


