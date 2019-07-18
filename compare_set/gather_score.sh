#!/bin/bash

. /u/local/Modules/default/init/modules.sh
module load python/3.7.2

cd /u/scratch/d/datduong/GOmultitask/compare_set

gapSize=300
server='/u/scratch/d/datduong'

pair='FlyWorm'
dataDir=$server/'geneOrtholog/'$pair'Score/qnliFormatData17'


for output in cosine.768.reduce300ClsVec ; do # cosine.768.reduce300ClsVec GcnRelu300Cosine cosine.bilstm.300Vec 

  finalDir=$server/'geneOrtholog/'$pair'Score/qnliFormatData17/'$output

  for point in {0..3600..300}
  do 

    echo $point
    scoreFile=$finalDir/'score.'$point'.txt'
    nameOut=$finalDir/'finalscore.'$point'.txt'
    pickleDf=$dataDir/'GeneDict2test.'$point'.pickle'
    # pickleDf,scoreFile,nameOut,start
    python3 gather_score.py $pickleDf $scoreFile $nameOut $point

  done
  cat $finalDir/finalscore*.txt > $finalDir/'GeneScore.txt'

done

