
import re,sys,os,pickle
import numpy as np

## convert output form 
# GO_xyz [ num ... ] into pickle 

os.chdir("/u/scratch/d/datduong/Onto2Vec/GOVectorData/2017")

output = {}
fin = open("VecResults_768.txt","r")
for line in fin: 
  line = re.sub(r"(\[|\])","",line.strip())
  line = line.split()
  goName = re.sub("GO_","GO:", line[0]) 
  if len(line[1::]) != 768 : 
    print (goName)
    continue
  output[goName] = np.array ( [ float (l) for l in line[1::] ] ) 


fin.close() 
pickle.dump(output,open("onto2vec2017dim768.pickle","wb"))
print (len(output))
