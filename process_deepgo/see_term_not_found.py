
## deepgo uses 2016 data, so we may not see all recent go terms ? 

from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata, string, re, sys, os, pickle, pickle, gzip

import numpy as np
import pandas as pd 

from copy import deepcopy

import networkx
import obonet

## using the COO format of sklearn 

work_dir = '/u/flashscratch/d/datduong/goAndGeneAnnotationDec2018/'
os.chdir (work_dir)

# Read the taxrank ontology
graph = obonet.read_obo('go.obo') # https://github.com/dhimmel/obonet


full_label_name_array = pd.read_csv("go_name_in_obo.csv", header=None)

all_name_array_deepgo = pd.read_csv("/u/flashscratch/d/datduong/deepgo/data/train/deepgo.mf.csv", header=None)

full_label_name_array = list (full_label_name_array[0])
all_name_array_deepgo = list (all_name_array_deepgo[0])

not_found = [x for x in all_name_array_deepgo if x not in full_label_name_array]

# m = ['GO:0000982', 'GO:0001133', 'GO:0001200', 'GO:0001201', 'GO:0001202', 'GO:0001203', 'GO:0003705']
# [k for k in m if k in all_name_array_deepgo]
# ['GO:0000982', 'GO:0003705']

