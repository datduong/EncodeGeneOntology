


SCRIPT = """
#!/bin/bash
. /u/local/Modules/default/init/modules.sh

# where_data='/u/scratch/d/datduong/deepgo/data/train/fold_1'
# cd $where_data
# set_type='test' ## what do we test on, test/dev set ?? 
# for category in ONTOTYPE ; do 

#   result_dir=$where_data/'blastPsiblastResultEvalEVALUE'
#   mkdir $result_dir

#   file_to_test=$set_type'-'$category'.fasta'

#   ## blast
#   fout2=$set_type'-'$category'.blast.txt'
#   /u/flashscratch/d/datduong/ncbi-blast-2.8.1+/bin/blastp -db train-$category.fasta -query $file_to_test -outfmt 10 -evalue EVALUE -out $result_dir/$fout2
#   sed -i 's/Search has CONVERGED!//g' $result_dir/$fout2

#   ## psiblast
#   fout1=$set_type'-'$category'.psiblast.txt'
#   /u/flashscratch/d/datduong/ncbi-blast-2.8.1+/bin/psiblast -num_threads 4 -db train-$category.fasta -query $file_to_test -num_iterations 3 -outfmt 10 -evalue EVALUE -out $result_dir/$fout1 
#   ## MUST REMOVE THE SINGLE LINE "Search has CONVERGED!"
#   sed -i 's/Search has CONVERGED!//g' $result_dir/$fout1

# done 


module load python/3.7.2
main_dir='/u/scratch/d/datduong/deepgo/data/train/'
where_data=$main_dir/'fold_1/'
set_type='test' ## what do we test on, test/dev set ?? 
code_dir='/u/scratch/d/datduong/GOmultitask/blastp/'
cd $code_dir
for evalpoint in EVALUE ; do # '1' '0.1' '0.01' '0.001'
  result_dir=$where_data/'blastPsiblastResultEval'$evalpoint'/'
  for category in ONTOTYPE ; do # bp mf
    all_test_label=$main_dir'deepgo'.$category.csv
    label_subset_file=$main_dir/'deepgo.'$category'.csv' ## these are the GO terms to be tested 
    blast=$result_dir/$set_type'-'$category'.blast.txt'
    psiblast=$result_dir/$set_type'-'$category'.psiblast.txt'
    python3 run_blast_psiblast.py $main_dir $where_data $result_dir $set_type $category $all_test_label > $result_dir/$category.output.txt
  done
done


"""

import os,sys,re,pickle 

os.chdir('/u/scratch/d/datduong/deepgo/data/train/fold_1')
counter = 1
for evalue in ['1','10','100']: 
  for onto in ['mf','cc','bp']: 
    script = re.sub('ONTOTYPE',onto,SCRIPT)
    script = re.sub('EVALUE',evalue,script)
    fout = open('RunBlast'+str(counter)+'.sh','w')
    fout.write(script)
    fout.close()
    os.system ( 'qsub -l h_data=4G,highp,h_rt=23:50:50 -pe shared 3 ' + 'RunBlast'+str(counter)+'.sh')
    counter = counter + 1 

