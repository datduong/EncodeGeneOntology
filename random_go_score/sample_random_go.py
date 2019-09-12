



from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata, string, re, sys, os, pickle, pickle, gzip

from tqdm import tqdm

import numpy as np
import pandas as pd

from copy import deepcopy

import networkx
import obonet

sys.path.append('/u/flashscratch/d/datduong/GOmultitask')
from compare_set import compare_set

work_dir = '/u/flashscratch/d/datduong/goAndGeneAnnotationMar2017/'
os.chdir (work_dir)

# Read the taxrank ontology
graph = obonet.read_obo('go.obo') # https://github.com/dhimmel/obonet

go_name_array = pd.read_csv("go_name_in_obo.csv",header=None)
go_name_array = list ( go_name_array[0] )

GeneGOdb = compare_set.def2dict("go_def_in_obo.tsv")

"""
sample 10000 random GO pairs, for each species, compare the method.
IC method will have significant difference, but definition model will have the same similarity
"""

np.random.seed(1)

random_len = 15100 ## it was 5100 should we do more ?? 

short_name = {"mf":'molecular_function', "bp": 'biological_process', "cc":'cellular_component'}

for category in ['mf','bp','cc'] :

  category_name = short_name[category]
  GO_in_category = [ g for g in go_name_array if graph.node[g]['namespace'] == category_name ]

  ## sample random
  set1 = np.random.choice(GO_in_category, size=random_len, replace=True)
  set2 = np.random.choice(GO_in_category, size=random_len, replace=True)

  ## filter set
  random_set = open("random_go_analysis_"+category+".tsv","w")
  random_set.write("index\tquestion\tsentence\tgo1\tgo2\tlabel\ttype")

  already_seen = {} ## don't record what we have seen
  counter = 0

  for pair in tqdm ( zip (set1,set2) ):
    if pair[0] == pair[1]:
      continue ## same node, so skip

    if pair[0]+pair[1] in already_seen:
      continue

    is_parent_child = False

    ## check they're parent and children
    for child, parent, key in graph.out_edges(pair[0], keys=True):
      if parent == pair[1]: ## node1 is parent of node0
        is_parent_child = True
        break

    for child, parent, key in graph.out_edges(pair[1], keys=True):
      if parent == pair[0]: ## node0 is parent of node1
        is_parent_child = True
        break

    if is_parent_child:
      continue

    ## write out random pairs
    # print (pair[0])
    # print (GeneGOdb[pair[0]]) ## why does it give a list ??

    random_set.write("\n"+str(counter)+'\t'+GeneGOdb[pair[0]][0] + "\t"+ GeneGOdb[pair[1]][0] +"\t"+pair[0]+"\t"+pair[1]+"\tnot_entailment\t"+ category)

    already_seen[pair[0]+pair[1]] = counter
    counter = counter + 1


  random_set.close()


  ### ****

  # ## true parent-child pairs

  # ## filter set
  # related_set = open("ParentChild_go_analysis_"+category+".tsv","w")
  # related_set.write("index\tquestion\tsentence\tgo1\tgo2\tlabel\ttype")

  # already_seen = {}

  # counter = 0
  # for node in tqdm ( GO_in_category ):

  #   ## get parent and children
  #   parents = []
  #   for child, parent, key in graph.out_edges(node, keys=True):
  #     # if key == 'is_a':
  #     parents.append(parent)

  #   if len(parents) == 0:
  #     continue

  #   size = len(parents)//2
  #   if size == 0:
  #     size = 1

  #   parents = np.random.choice( parents,size=size ) ## down sample a little bit

  #   parents = [p for p in parents if (node+p) not in already_seen]

  #   for p in parents:
  #     ## write out
  #     related_set.write("\n"+str(counter)+'\t'+GeneGOdb[node][0]+"\t"+GeneGOdb[p][0]+"\t"+node+"\t"+p+"\tentailment\t"+ category)

  #     already_seen[node+p] = counter
  #     counter = counter + 1

  # related_set.close()







