
import os,sys,re,pickle
import numpy as np 


script = """

#!/bin/bash

. /u/local/Modules/default/init/modules.sh
module load python/3.7.2

cd /u/scratch/d/datduong/GOmultitask/compare_set

server='/u/scratch/d/datduong'

pair=PAIR
dataDir=$server/'geneOrtholog/'$pair'Score/qnliFormatData17'


for output in BertGOName768 ; do 

  finalDir=$server/'geneOrtholog/'$pair'Score/qnliFormatData17/'$output

  for point in {0..ENDPOINT..GAP}
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


"""

os.chdir('/u/scratch/d/datduong/geneOrtholog')

# pair={'Human':[5300,50],'Yeast':[11700,300]}

pair={'HumanMouse':[22500,300],  
      'HumanFly':[10500,300], 
      'MouseFly':[12600,300], 
      'FlyWorm':[3600,300] }

fout = open("run_gather_score5.sh","w")
for key,val in pair.items() : 
  script2 = re.sub("PAIR", "'"+key+"'", script)
  script2 = re.sub("ENDPOINT", str(val[0]) ,script2)
  script2 = re.sub("GAP", str(val[1]) ,script2)
  fout.write(script2+ "\n\n\n")


fout.close() 


