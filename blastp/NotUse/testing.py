import pickle 
z = pickle.load(open("/u/scratch/d/datduong/deepgo/data/train/fold_1/blastPsiblastResult/test-cc.dict.pickle","rb"))
w = pickle.load(open("/u/scratch/d/datduong/deepgo/data/train/fold_1/test-cc.TrueLabel.pickle","rb"))

notfound = []
for k in w: 
  if k not in z: ## small set
    notfound.append(k)


'P52052'
