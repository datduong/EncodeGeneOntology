


SCRIPT = """

#!/bin/bash
. /u/local/Modules/default/init/modules.sh

where_data='/u/scratch/d/datduong/deepgo/dataExpandGoSet16Jan2020/train/fold_1'
cd $where_data
set_type='test' ## what do we test on, test/dev set ??
for category in ONTOTYPE ; do

  result_dir=$where_data/'blastPsiblastResultEvalEVALUE'
  mkdir $result_dir

  file_to_test=$set_type'-'$category'-same-origin.fasta'

  ##!! blast
  fout2=$set_type'-'$category'.blast.txt'
  /u/flashscratch/d/datduong/ncbi-blast-2.8.1+/bin/blastp -db train-$category-same-origin.fasta -query $file_to_test -outfmt 10 -evalue EVALUE -out $result_dir/$fout2
  sed -i 's/Search has CONVERGED!//g' $result_dir/$fout2

  ##!! psiblast
  fout1=$set_type'-'$category'.psiblast.txt'
  /u/flashscratch/d/datduong/ncbi-blast-2.8.1+/bin/psiblast -num_threads 4 -db train-$category-same-origin.fasta -query $file_to_test -num_iterations 3 -outfmt 10 -evalue EVALUE -out $result_dir/$fout1
  ##!! MUST REMOVE THE SINGLE LINE "Search has CONVERGED!"
  sed -i 's/Search has CONVERGED!//g' $result_dir/$fout1
done


#### add in the option add_name for larger dataset.
add_name='-same-origin'


module load python/3.7.2
main_dir='/u/scratch/d/datduong/deepgo/dataExpandGoSet16Jan2020/train/'
where_data=$main_dir/'fold_1/'
set_type='test' ## what do we test on, test/dev set ??
code_dir='/u/scratch/d/datduong/GOmultitask/blastp/'
cd $code_dir
for evalpoint in EVALUE ; do
  result_dir=$where_data/'blastPsiblastResultEval'$evalpoint'/'
  for category in ONTOTYPE ; do
    all_test_label=$main_dir'deepgo'.$category.csv
    label_subset_file=$main_dir/'deepgo.'$category'.csv' ## these are the GO terms to be tested
    blast=$result_dir/$set_type'-'$category'.blast.txt'
    psiblast=$result_dir/$set_type'-'$category'.psiblast.txt'
    python3 run_blast_psiblast.py $main_dir $where_data $result_dir $set_type $category $all_test_label $add_name > $result_dir/$category.output.txt
  done
done



"""

import os,sys,re,pickle

os.chdir('/u/scratch/d/datduong/deepgo/dataExpandGoSet16Jan2020/train/fold_1')
counter = 1
for evalue in ['10','100']:
  for onto in ['mf','cc','bp']:
    script = re.sub('ONTOTYPE',onto,SCRIPT)
    script = re.sub('EVALUE',evalue,script)
    fout = open('RunBlast'+str(counter)+'.sh','w')
    fout.write(script)
    fout.close()
    os.system ( 'qsub -l h_data=8G,highp,h_rt=16:50:50 -pe shared 2 ' + 'RunBlast'+str(counter)+'.sh')
    counter = counter + 1

