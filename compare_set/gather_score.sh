#!/bin/bash

. /u/local/Modules/default/init/modules.sh
module load python/3.7.2

cd /u/scratch/d/datduong/GOmultitask/compare_set

gapSize=300
server='/u/scratch/d/datduong'

pair='HumanFly'
dataDir=$server/'geneOrtholog/'$pair'Score/qnliFormatData17'


for output in cosine.bilstm.300Vec ; do # cosine.768.reduce300ClsVec GcnRelu300Cosine cosine.bilstm.300Vec 

  finalDir=$server/'geneOrtholog/'$pair'Score/qnliFormatData17/'$output

  for point in {0..10500..300}
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



## ****

#!/bin/bash

. /u/local/Modules/default/init/modules.sh
module load python/3.7.2

cd /u/scratch/d/datduong/GOmultitask/compare_set

server='/u/scratch/d/datduong'

pair='Human'
dataDir=$server/$pair'PPI3ontology/qnliFormatData17'


for output in cosine.bilstm.300Vec GcnRelu300Cosine cosine.768.reduce300ClsVec ; do # cosine.768.reduce300ClsVec GcnRelu300Cosine cosine.bilstm.300Vec 

  finalDir=$dataDir/$output

  for point in {0..5300..50} # 11700
  do 

    echo $point
    scoreFile=$finalDir/'score.'$point'.txt'
    nameOut=$finalDir/'finalscore.'$point'.txt'
    pickleDf=$dataDir/'GeneDict2test.'$point'.pickle'
    # pickleDf,scoreFile,nameOut,start
    python3 gather_score.py $pickleDf $scoreFile $nameOut $point

  done
  cat $finalDir/finalscore*.txt > $finalDir/'GeneScore.tsv'

done



. /u/local/Modules/default/init/modules.sh
module load python/3.7.2

cd /u/scratch/d/datduong/GOmultitask/compare_set

server='/u/scratch/d/datduong'

pair='Yeast'
dataDir=$server/$pair'PPI3ontology/qnliFormatData17'


for output in cosine.bilstm.300Vec GcnRelu300Cosine cosine.768.reduce300ClsVec ; do # cosine.768.reduce300ClsVec GcnRelu300Cosine cosine.bilstm.300Vec 

  finalDir=$dataDir/$output

  for point in {0..11700..300} # 11700
  do 

    echo $point
    scoreFile=$finalDir/'score.'$point'.txt'
    nameOut=$finalDir/'finalscore.'$point'.txt'
    pickleDf=$dataDir/'GeneDict2test.'$point'.pickle'
    # pickleDf,scoreFile,nameOut,start
    python3 gather_score.py $pickleDf $scoreFile $nameOut $point

  done
  cat $finalDir/finalscore*.txt > $finalDir/'GeneScore.tsv'

done


