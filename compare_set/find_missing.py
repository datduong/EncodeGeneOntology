
import pandas as pd 

df1 = pd.read_csv ( "/u/scratch/d/datduong/geneOrtholog/FlyWormScore/FlyWormOrtholog2TestTrim.txt", dtype=str, sep=' ') 

df2 = pd.read_csv ( "/u/scratch/d/datduong/geneOrtholog/FlyWormScore/qnliFormat/GcnRelu300Cosine/GeneScore.txt", dtype=str, sep=',') 
df2.columns = ['gene1','gene2','label','score1','score2']

largePair = [(df1['gene1'][i], df1['gene2'][i]) for i in range(0,df1.shape[0]) ]

smallPair = [(df2['gene1'][i], df2['gene2'][i]) for i in range(0,df2.shape[0]) ]

missing = [i for i in largePair if i not in smallPair]
('CG14937', 'C46C2.2')
