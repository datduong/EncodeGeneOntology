

from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata, string, re, sys, os, pickle, pickle, gzip

import numpy as np
import pandas as pd 

from copy import deepcopy

import networkx
import obonet

## using the COO format of sklearn 

## deepgo/data/train

# work_dir = '/u/flashscratch/d/datduong/goAndGeneAnnotationDec2018/'
# work_dir = '/u/flashscratch/d/datduong/goAndGeneAnnotation/'
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


go_name_array_obo = list(id_to_name.keys())
go_name_array_obo.sort() 
# go_name_array_obo = [re.sub(r"GO:","",g) for g in go_name_array_obo]

pd.DataFrame(go_name_array_obo).to_csv("go_name_in_obo.csv", header=None, index=None)

# graph.node['GO:0000002']['def']

# get def from obo file 
# split by quote ""
# [Term]
# id: GO:0000001
# name: mitochondrion inheritance
# namespace: biological_process
# def: "The distribution of mitochondria, including the mitochondrial genome, into daughter cells after mitosis or meiosis, mediated by interactions between mitochondria and the cytoskeleton." [GOC:mcc, PMID:10873824, PMID:11389764]
# synonym: "mitochondrial inheritance" EXACT []
# is_a: GO:0048308 ! organelle inheritance
# is_a: GO:0048311 ! mitochondrion distribution

fout = open ("go_def_in_obo.tsv","w")
fout.write("name\tdef")
for node_name in go_name_array_obo: 
  node_def = graph.node[node_name]['def'].split('"')[1] # split by quote, take 2nd entry
  defin = graph.node[node_name]['name'] + " " + node_def
  defin = re.sub(r"[\n\t]", " ", defin)
  defin = defin.strip() 
  fout.write("\n"+node_name + "\t" + defin.lower())


fout.close()

# Find all superterms of species. Note that networkx.descendants gets
# superterms, while networkx.ancestors returns subterms.
# networkx.descendants(graph, 'GO:0000002') ## return parents https://www.ebi.ac.uk/QuickGO/term/GO:0000002

# the first list contains the index of the source nodes, 
# while the index of target nodes is specified in the second list.
# edge_index = torch.tensor([[0, 1, 2, 0, 3],
#                            [1, 0, 1, 3, 2]], dtype=torch.long)


edge_index_source=[]
edge_index_target=[]

# https://github.com/dhimmel/obonet/blob/master/examples/go-obonet.ipynb
for node_name in go_name_array_obo: 
  # node_name = 'GO:0033955'
  source_index = go_name_array_obo.index(node_name)
  for child, parent, key in graph.out_edges(node_name, keys=True):
    # @key tells "is_a" or "part_of" ...
    edge_index_source.append ( source_index )
    edge_index_target.append ( go_name_array_obo.index(parent) ) ## the parents of this node


#
edge_index = np.array ( [ edge_index_source, edge_index_target] ) 

pickle.dump ( edge_index , open("adjacency_matrix_coo_format.pickle","wb") )


