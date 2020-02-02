import sys,re,pickle,os
import pandas as pd
import numpy as np

#### make fake vectors to test model

name = []
num_go_label = 0
fin = open('/local/datdb/deepgo/data/go_def_in_obo.tsv','r')
for index, line in enumerate(fin):
  if index == 0: ## 1st line.
    continue
  name.append(line.split()[0])
  num_go_label = 1 + num_go_label

print ("\n num label {}".format(num_go_label))

label_dim = 768
# value = np.random.randn(num_go_label,label_dim) * 0.25 ## scale down a little
value = np.zeros((num_go_label,label_dim)) * 0.25 ## scale down a little
value = np.random.uniform(low=-1,high=1,size=(num_go_label,label_dim))


## create dictionary output
label_vec={}
for index,go in enumerate(name):
  label_vec[go] = value[index]

#
os.mkdir('/local/datdb/deepgo/data/UniformGOVector')
os.chdir('/local/datdb/deepgo/data/UniformGOVector')
pickle.dump(label_vec,open("label_vector.pickle",'wb'))

