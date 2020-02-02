
import os,sys,re,pickle

#### convert GO vector text file into dictionary

os.chdir("/local/datdb/deepgo/data/BertAsServiceVector")

fin = open("bertASdeepgo.csv","r")
GOcount = 0
dim = 0
go_dict = {}
for index,line in enumerate(fin):
  line = line.strip().split()
  if index==0: ## first line
    # GOcount = int(line[0])
    dim = 768
  else:
    if index == 1: 
      print ('\ndim {}'.format(len(line[1::])))
    go_dict[line[0]] = [ float(num) for num in line[1::] ] ## convert to np ?? for now... let's no do it.

#
pickle.dump( go_dict, open("label_vector.pickle","wb") )
print ('\ntotal number of go labels {}'.format(len(go_dict)))
