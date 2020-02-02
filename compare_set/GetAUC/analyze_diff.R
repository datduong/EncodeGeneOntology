
tail (dataUsedInAuc[c('g1','g2','aicMean','meanBMA')], 40)

df0 = subset ( dataUsedInAuc, dataUsedInAuc$label==0 )
df1 = subset ( dataUsedInAuc, dataUsedInAuc$label==1 )

summary ( df0['aicMean'] ) 
summary ( (df0['meanBMA'] + 1) / 2 ) 


summary ( df1['aicMean'] ) 
summary ( (df1['meanBMA'] + 1) / 2 ) 



Olfr320    stac

Olfr320 GO:0004984
Olfr320 GO:0007186 **
Olfr320 GO:0007608
Olfr320 GO:0016021

stac    GO:0003674
stac    GO:0035147 **
stac    GO:0099503

cd /u/scratch/d/datduong/geneOrtholog/MouseFlyScore/qnliFormatData17
grep -P "GO:0007186\tGO:0035147" GcnRelu300Cosine/score*txt
grep -P "GO:0007186\tGO:0035147" cosine.768.reduce300ClsVec/score*txt



Fndc11          CG15579 0.07449035 0.8183507

Fndc11  GO:0070062 **
Fndc11  GO:0008150
Fndc11  GO:0003674

CG15579 GO:0008150
CG15579 GO:0003674
CG15579 GO:0005575 **


cd /u/scratch/d/datduong/geneOrtholog/MouseFlyScore/qnliFormatData17
grep -P "GO:0070062\tGO:0005575" GcnRelu300Cosine/score*txt
grep -P "GO:0070062\tGO:0005575" cosine.768.reduce300ClsVec/score*txt




Timm13          CG12099 0.12745363 0.4961830
Timm13  GO:0001650 ** cc
Timm13  GO:0005739
Timm13  GO:0008565
Timm13  GO:0042719
Timm13  GO:0045039 ** bp 
Timm13  GO:0072321

CG12099 GO:0005575
CG12099 GO:0004842
CG12099 GO:0051865 **
CG12099 GO:0008270

cd /u/scratch/d/datduong/geneOrtholog/MouseFlyScore/qnliFormatData17
grep -P "GO:0045039\tGO:0051865" GcnRelu300Cosine/score*txt
grep -P "GO:0045039\tGO:0051865" cosine.768.reduce300ClsVec/score*txt

grep -P "GO:0045039\tGO:0051865" GcnRelu300Cosine/score*txt
grep -P "GO:0045039\tGO:0051865" cosine.768.reduce300ClsVec/score*txt



x = read.table('/u/scratch/d/datduong/geneOrtholog/MouseFlyScore/qnliFormatData17/GcnRelu300Cosine/score.9900.degree.txt',sep="\t",header=TRUE)
xOver0 = subset (x, x$score<1 & x$score>0)
summary ( xOver0$degree1 ) 
summary ( xOver0$degree2 ) 
summary ( xOver0$MeanDegree ) 

xUnder0 = subset (x, x$score<0)
summary ( xUnder0$degree1 ) 
summary ( xUnder0$degree2 ) 
summary ( xUnder0$MeanDegree ) 

# so there is no major differences between distribution of degrees when consider all 3 ontology jointly, but why does GCN does bad? We can cherry pick?

cor(x$score,x$MeanDegree)



xHighDegree=subset (x, x$MeanDegree>20)
cor(xHighDegree$score,xHighDegree$MeanDegree)

## look at high level term, large degree, pick that node, and its children, see what the scores are. 