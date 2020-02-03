

#### Compare 2 proteins by measuring the sim score for their sets of GO terms annotating them.

import string, re, sys, os, pickle, gzip
import numpy as np
from scipy.spatial import distance

Label4Prot1 = 'GO:0005634; GO:0005654; GO:0005737; GO:0008284; GO:0016604; GO:0030054; GO:0043231; GO:0045892; GO:0051260'.split(';')
Label4Prot2 = 'GO:0005654; GO:0005829; GO:0006361; GO:0006362; GO:0006363; GO:0032481; GO:0045815'.split(';')

GoVectorPath = 'path/to/your/vector'
GoVectorDict = pickle.load (open(GoVectorPath,'rb'))

ScoreMatrixPairWiseGO = np.zeros((len(Label4Prot1),len(Label4Prot2)))
Label4Prot1 = sorted(Label4Prot1)
Label4Prot2 = sorted(Label4Prot2)

#### compare every pairwise between 2 sets

for i1,GoIn1 in enumerate(Label4Prot1):
  for i2,GoIn2 in enumerate(Label4Prot2):
    ScoreMatrixPairWiseGO[i1,i2] = distance.cosine(GoVectorDict[GoIn1],GoVectorDict[GoIn2]) + 1 # we add 1 so that close vector has high score.


#### apply metric to compare sets

RowMax = np.amax(ScoreMatrixPairWiseGO,1) ## max for each row ... same as max_b s(a,b) 
RowMean = np.mean(RowMax) ## mean_a max_b s(a,b)
ColMax = np.amax(ScoreMatrixPairWiseGO,0) ## max for each col
ColMean = np.mean(ColMax)
print ( 'score is {}'.format(np.mean([RowMean,ColMean])) ) # hausdorff distance ... same as best-match average




