#!/bin/bash

. /u/local/Modules/default/init/modules.sh
module load python/3.7.2

cd /u/scratch/d/datduong/GOmultitask/compare_set

gapSize=300
server='/u/scratch/d/datduong'

dataDir=$server/'geneOrtholog/HumanMouseScore/qnliFormat'

# finalDir=$server/'geneOrtholog/HumanMouseScore/qnliFormat/GcnRelu300Cosine'

finalDir=$server/'geneOrtholog/HumanMouseScore/qnliFormat/cosine.768.reduce300ClsVec'


# 22800
for point in {0..22500..300}
do 

echo $point
scoreFile=$finalDir/'score.'$point'.txt'
nameOut=$finalDir/'finalscore.'$point'.txt'
pickleDf=$dataDir/'GeneDict2test.'$point'.pickle'
# pickleDf,scoreFile,nameOut,start
python3 gather_score.py $pickleDf $scoreFile $nameOut $point

done

cat $finalDir/finalscore*.txt > $finalDir/'GeneScore.txt'
