#!/bin/bash

. /u/local/Modules/default/init/modules.sh
module load python/3.7.2

cd /u/scratch/d/datduong/GOmultitask/compare_set

# 54178
gapSize=300
server='/u/scratch/d/datduong'
genePairList=$server/'geneOrtholog/HumanMouseScore/HumanMouseOrtholog2TestTrim.txt'
outDir=$server/'geneOrtholog/HumanMouseScore/qnliFormat'
mkdir $outDir

gaf1='goa_human_not_IEA.tsv'
gaf2='mgi_not_IEA.tsv'

wc -l $genePairList

for point in {0..22500..300}
do 

echo $point
saveDf=$outDir/'Ortholog2testDef.'$point'.txt'
savePickle=$outDir/'GeneDict2test.'$point'.pickle'
python3 process_data.py $gaf1 $gaf2 $saveDf $savePickle $genePairList $point $gapSize

done

