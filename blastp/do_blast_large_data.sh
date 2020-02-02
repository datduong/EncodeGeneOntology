
## make fasta database 
where_data='/u/scratch/d/datduong/deepgo/dataExpandGoSet16Jan2020/train/fold_1'
cd $where_data
for category in cc bp mf ; do 
  /u/flashscratch/d/datduong/ncbi-blast-2.8.1+/bin/makeblastdb -in train-$category-same-origin.fasta -dbtype prot -parse_seqids
done


#### do psiblast and blast 
#!/bin/bash
. /u/local/Modules/default/init/modules.sh
where_data='/u/scratch/d/datduong/deepgo/dataExpandGoSet16Jan2020/train/fold_1'
cd $where_data
set_type='test' ## what do we test on, test/dev set ?? 
for category in bp ; do 

  result_dir=$where_data/'blastPsiblastResultEval10'
  mkdir $result_dir

  file_to_test=$set_type'-'$category'-same-origin.fasta'

  ##!! blast
  fout2=$set_type'-'$category'.blast.txt'
  /u/flashscratch/d/datduong/ncbi-blast-2.8.1+/bin/blastp -db train-$category-same-origin.fasta -query $file_to_test -outfmt 10 -evalue 10 -out $result_dir/$fout2
  sed -i 's/Search has CONVERGED!//g' $result_dir/$fout2

  ##!! psiblast
  fout1=$set_type'-'$category'.psiblast.txt'
  /u/flashscratch/d/datduong/ncbi-blast-2.8.1+/bin/psiblast -num_threads 4 -db train-$category-same-origin.fasta -query $file_to_test -num_iterations 3 -outfmt 10 -evalue 10 -out $result_dir/$fout1 
  ##!! MUST REMOVE THE SINGLE LINE "Search has CONVERGED!"
  sed -i 's/Search has CONVERGED!//g' $result_dir/$fout1
done 

#### tally outcome of blast and psiblast
#!/bin/bash
. /u/local/Modules/default/init/modules.sh
module load python/3.7.2
main_dir='/u/scratch/d/datduong/deepgo/dataExpandGoSet16Jan2020/train/'
where_data=$main_dir/'fold_1/' # /u/scratch/d/datduong/deepgo/dataExpandGoSet16Jan2020/train/fold_1/test-mf.tsv
set_type='test' ## what do we test on, test/dev set ?? 
code_dir='/u/scratch/d/datduong/GOmultitask/blastp/'
cd $code_dir
for evalpoint in '1' '10' '100' ; do # '1' '0.1' '0.01' '0.001'
  result_dir=$where_data/'blastPsiblastResultEval'$evalpoint'/'
  for category in bp cc ; do # bp mf
    all_test_label=$main_dir'deepgo'.$category.csv
    label_subset_file=$main_dir/'deepgo.'$category'.csv' ## these are the GO terms to be tested 
    blast=$result_dir/$set_type'-'$category'.blast.txt'
    psiblast=$result_dir/$set_type'-'$category'.psiblast.txt'
    python3 run_blast_psiblast.py $main_dir $where_data $result_dir $set_type $category $all_test_label > $result_dir/$category.output.txt
  done
done





