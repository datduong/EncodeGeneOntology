#!/bin/bash

. /u/local/Modules/default/init/modules.sh
module load python/3.7.2

server='/u/scratch/d/datduong'

previousSource=$server/'goAndGeneAnnotationMar2017'/'gafData2017'

gaf1=$previousSource/'mgi_not_IEA.tsv' #goa_human_not_IEA
gaf2=$previousSource/'fb_not_IEA.tsv'

pair='MouseFly'
genePairList=$server/'geneOrtholog/'$pair'Score/'$pair'Ortholog2TestTrim.txt'
outDir=$server/'geneOrtholog/'$pair'Score/qnliFormatData17'
mkdir $outDir

wc -l $genePairList

gapSize=300
for point in {0..12900..300} # 3900 22800 10800 12600
do 

echo $point
saveDf=$outDir/'Ortholog2testDef.'$point'.txt'
savePickle=$outDir/'GeneDict2test.'$point'.pickle'
cd /u/scratch/d/datduong/GOmultitask/compare_set
python3 process_data.py $gaf1 $gaf2 $saveDf $savePickle $genePairList $point $gapSize

done


## **** ## **** ## ****
# we can use the same code to get prot interaction pairs in the same species 

#!/bin/bash
. /u/local/Modules/default/init/modules.sh
module load python/3.7.2

server='/u/scratch/d/datduong'

previousSource=$server/'goAndGeneAnnotationMar2017'/'gafData2017'

gaf1=$previousSource/'sgd_not_IEA.tsv' #goa_human_not_IEA

pair='Yeast'

genePairList=$server/$pair'PPI3ontology/'$pair'PPI2TestTrim.txt'
outDir=$server/$pair'PPI3ontology/qnliFormatData17'

mkdir $outDir

wc -l $genePairList

gapSize=300
for point in {0..12000..300} # 5350
do 

echo $point
saveDf=$outDir/'PPI2testDef.'$point'.txt'
savePickle=$outDir/'GeneDict2test.'$point'.pickle'

## use the same annotation file for both input
cd /u/scratch/d/datduong/GOmultitask/compare_set
python3 process_data.py $gaf1 $gaf1 $saveDf $savePickle $genePairList $point $gapSize

done



## **** ## **** ## ****
# we can use the same code to get prot interaction pairs in the same species 

#!/bin/bash
. /u/local/Modules/default/init/modules.sh
module load python/3.7.2

server='/u/scratch/d/datduong'

previousSource=$server/'goAndGeneAnnotationMar2017'/'gafData2017'

gaf1=$previousSource/'human_not_IEA.tsv' #goa_human_not_IEA

pair='Human'

genePairList=$server/$pair'PPI3ontology/'$pair'PPI2TestTrim.txt'
outDir=$server/$pair'PPI3ontology/qnliFormatData17'

mkdir $outDir

wc -l $genePairList

gapSize=50
for point in {0..5350..50} # 5350
do 

echo $point
saveDf=$outDir/'PPI2testDef.'$point'.txt'
savePickle=$outDir/'GeneDict2test.'$point'.pickle'

## use the same annotation file for both input
cd /u/scratch/d/datduong/GOmultitask/compare_set
python3 process_data.py $gaf1 $gaf1 $saveDf $savePickle $genePairList $point $gapSize

done

