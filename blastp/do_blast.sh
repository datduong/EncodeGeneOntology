
## make fasta database 
where_data='/u/scratch/d/datduong/deepgo/dataExpandGoSet/train/fold_1'
cd $where_data
for category in cc bp mf ; do 
  /u/flashscratch/d/datduong/ncbi-blast-2.8.1+/bin/makeblastdb -in train-$category.fasta -dbtype prot -parse_seqids
done


#### do psiblast and blast 
#!/bin/bash
. /u/local/Modules/default/init/modules.sh
where_data='/u/scratch/d/datduong/deepgo/data/train/fold_1'
cd $where_data
set_type='test' ## what do we test on, test/dev set ?? 
for category in mf ; do 

  result_dir=$where_data/'blastPsiblastResultEval10'
  mkdir $result_dir

  file_to_test=$set_type'-'$category'.fasta'

  ##!! blast
  fout2=$set_type'-'$category'.blast.txt'
  /u/flashscratch/d/datduong/ncbi-blast-2.8.1+/bin/blastp -db train-$category.fasta -query $file_to_test -outfmt 10 -evalue 10 -out $result_dir/$fout2
  sed -i 's/Search has CONVERGED!//g' $result_dir/$fout2

  ##!! psiblast
  fout1=$set_type'-'$category'.psiblast.txt'
  /u/flashscratch/d/datduong/ncbi-blast-2.8.1+/bin/psiblast -num_threads 4 -db train-$category.fasta -query $file_to_test -num_iterations 3 -outfmt 10 -evalue 10 -out $result_dir/$fout1 
  ##!! MUST REMOVE THE SINGLE LINE "Search has CONVERGED!"
  sed -i 's/Search has CONVERGED!//g' $result_dir/$fout1
done 

#### tally outcome of blast and psiblast
#!/bin/bash
. /u/local/Modules/default/init/modules.sh
module load python/3.7.2
main_dir='/u/scratch/d/datduong/deepgo/data/train/'
where_data=$main_dir/'fold_1/' # /u/scratch/d/datduong/deepgo/dataExpandGoSet/train/fold_1/test-mf.tsv
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







## !!! combine blast with BERT/biLSTM/GCN

#!/bin/bash
. /u/local/Modules/default/init/modules.sh
module load python/3.7.2

main_dir='/u/scratch/d/datduong/deepgo/data/train/'

where_data=$main_dir/'fold_1/'
set_type='test' ## what do we test on, test/dev set ?? 

code_dir='/u/scratch/d/datduong/GOmultitask/blastp/'
cd $code_dir

result_dir=$where_data/'blastPsiblastResult/'

wordvec_file='/u/scratch/d/datduong/deepgo/data/cosine.768.reduce300ClsVec/label_vector.txt'

for category in cc bp mf ; do # bp mf

  label_subset_file='/u/scratch/d/datduong/deepgo/data/deepgo.'$category'.csv' ## these are the GO terms to be tested 
  # wordvec_file_small='none'
  wordvec_file_small='/u/scratch/d/datduong/deepgo/data/cosine.768.reduce300ClsVec/label_vector.deepgo.'$category'.txt' ## do not test against all GO database 
  blast=$result_dir/$set_type'-'$category'.blast.txt'
  psiblast=$result_dir/$set_type'-'$category'.psiblast.txt'

  python3 do_expand_set.py $main_dir $where_data $result_dir $set_type $category $wordvec_file $wordvec_file_small $label_subset_file > $result_dir/'output.'$category'.txt'

done







## make database 
where_data='human-yeast-ecoli-L50-500-IsA-full/ProtDataJan19w300Base'
cd /u/flashscratch/d/datduong/goAndGeneAnnotationDec2018/uniprot_top_species/$where_data
# /u/flashscratch/d/datduong/ncbi-blast-2.8.1+/bin/makeblastdb -in train_seq_has_rare_go.fasta -dbtype prot -parse_seqids

## do psiblast 
where_query='/u/flashscratch/d/datduong/goAndGeneAnnotationDec2018/uniprot_top_species/'$where_data
cd $where_query
set_type='train'
file=$set_type'_seq_has_rare_go'

## blast
fout2='blast_'$set_type'set_our'
/u/flashscratch/d/datduong/ncbi-blast-2.8.1+/bin/blastp -db train_seq_has_rare_go.fasta -query $where_query/$file.fasta -outfmt 10 -evalue 0.01 -out $fout2.out 
sed -i 's/Search has CONVERGED!//g' $fout2.out 

## psiblast
fout1='psiblast_'$set_type'set_our'
/u/flashscratch/d/datduong/ncbi-blast-2.8.1+/bin/psiblast -num_threads 2 -db train_seq_has_rare_go.fasta -query $where_query/$file.fasta -num_iterations 3 -outfmt 10 -out $fout1.out 
## MUST REMOVE THE SINGLE LINE "Search has CONVERGED!"
sed -i 's/Search has CONVERGED!//g' $fout1.out 


## tally the outcome 

. /u/local/Modules/default/init/modules.sh
module load python/3.7.2
where_train='uniprot_top_species/human-yeast-ecoli-L50-500-IsA-full/ProtDataJan19w300Base'
where_blast='uniprot_top_species/human-yeast-ecoli-L50-500-IsA-full/ProtDataJan19w300Base'
where_test='uniprot_top_species/human-yeast-ecoli-L50-500-IsA-full/ProtDataJan19w300Base'
add_name='_our'
set_type='train'
what_set=$set_type'set'
fout='human-yeast-ecoli-L50-500-IsA-full.'$set_type'.blast'

## MUST REMOVE THE SINGLE LINE "Search has CONVERGED!"

cd /u/flashscratch/d/datduong/GOmultitask/blastp 
python3 /u/flashscratch/d/datduong/GOmultitask/blastp/run_blast_psiblast.py $where_train $what_set $where_blast $add_name

add_name=$what_set'_our'
python3 /u/flashscratch/d/datduong/GOmultitask/blastp/convert_np.py $where_train $set_type $where_test $add_name 0 > $fout.txt





## make database 
cd /u/flashscratch/d/datduong/goAndGeneAnnotationDec2018/uniprot_top_species/full/redo_dec17_w2v17
/u/flashscratch/d/datduong/ncbi-blast-2.8.1+/bin/makeblastdb -in train_seq_has_rare_go.fasta -dbtype prot -parse_seqids

## do blast test on gold-standard MetaGO
cd /u/flashscratch/d/datduong/goAndGeneAnnotationDec2018/uniprot_top_species/full/redo_dec17_w2v17
/u/flashscratch/d/datduong/ncbi-blast-2.8.1+/bin/psiblast -db train_seq_has_rare_go.fasta -query /u/flashscratch/d/datduong/goAndGeneAnnotationDec2018/benchmark_cofactor_method/1079.testing/gold_standard/dev_seq.fasta -num_iterations 3 -outfmt 10 -out psiblast_devset_metaGO.out 
## run blast
/u/flashscratch/d/datduong/ncbi-blast-2.8.1+/bin/blastp -db train_seq_has_rare_go.fasta -query /u/flashscratch/d/datduong/goAndGeneAnnotationDec2018/benchmark_cofactor_method/1079.testing/gold_standard/dev_seq.fasta -outfmt 10 -evalue 0.01 -out blast_devset_metaGO.out 


## do blast on uniprot 
cd /u/flashscratch/d/datduong/goAndGeneAnnotationDec2018/uniprot_top_species/30_300/redo_dec17_w2v17
/u/flashscratch/d/datduong/ncbi-blast-2.8.1+/bin/psiblast -db train_seq_has_rare_go.fasta -query dev_seq_has_rare_go.fasta -num_iterations 3 -outfmt 10 -out psiblast_devset_our.out 
## run blast
/u/flashscratch/d/datduong/ncbi-blast-2.8.1+/bin/blastp -db train_seq_has_rare_go.fasta -query dev_seq_has_rare_go.fasta -outfmt 10 -evalue 0.01 -out blast_devset_our_eval.out 


## tally the outcome 

## do it on MetaGO data 
# where_train, where_test, add_name
. /u/local/Modules/default/init/modules.sh
module load python/3.7.2
cd /u/flashscratch/d/datduong/GOmultitask/blastp 
python3 run_blast_psiblast.py uniprot_top_species/full/redo_dec17_w2v17 _metaGO
python3 convert_np.py uniprot_top_species/full/redo_dec17_w2v17 benchmark_cofactor_method/1079.testing/gold_standard _metaGO 1 > redo_dec17_w2v17_blast_meta_GO.txt


## !! do it on our own test set 
# where_train, where_test, add_name
. /u/local/Modules/default/init/modules.sh
module load python/3.7.2
cd /u/flashscratch/d/datduong/GOmultitask/blastp 
python3 run_blast_psiblast.py uniprot_top_species/30_300/redo_dec17_w2v17 uniprot_top_species/30_300/redo_dec17_w2v17 _our
python3 convert_np.py uniprot_top_species/30_300/redo_dec17_w2v17 uniprot_top_species/30_300/redo_dec17_w2v17 _our 0 > redo_dec17_w2v17_blast_our.txt


## !! do it on our own Arabidopsis test set 
# where_train, where_test, add_name
. /u/local/Modules/default/init/modules.sh
module load python/3.7.2
cd /u/flashscratch/d/datduong/GOmultitask/blastp 
# where_train, where_blast, add_name
python3 run_blast_psiblast.py uniprot_top_species/full/redo_dec17_w2v17 ArabidopsisUniprot _arabidopsis
python3 convert_np.py uniprot_top_species/full/redo_dec17_w2v17 ArabidopsisUniprot _arabidopsis 0 > redo_dec17_w2v17_blast_arabidopsis.txt



## 
blastpgp -a 1 -F F -j 3 -b 3000 -e 1e-3 -h 1e-3 -d <path to non-redundant blast database> -I <path to fasta file containing exactly one target protein> -Q <path to output profile file>

cd /u/flashscratch/d/datduong/goAndGeneAnnotationDec2018/uniprot_top_species/full/

/u/flashscratch/d/datduong/ncbi-blast-2.8.1+/bin/psiblast -db train_seq_has_rare_go.fasta -out_pssm my_protein.ckp -evalue 0.01 -query one_seq_test.fasta -out_ascii_pssm ascii_mtx_file -out output_file -num_iterations 3

/u/flashscratch/d/datduong/ncbi-blast-2.8.1+/bin/psiblast -db train_seq_has_rare_go.fasta -evalue 0.01 -query two_seq_test.fasta -out_ascii_pssm ascii_mtx_file3 -out output_file -num_iterations 3 -num_threads 4


/local/chelseaju/software/ncbi-blast-2.7.1+/bin/psiblast  -evalue 1e-3 -num_alignments 3000 -db /local/chelseaju/Database/NR/nr -query seq.fasta -num_iterations 3 -out_ascii_pssm seq.ascii -out seq.out -num_threads 8


psiblast [-h] [-help] [-import_search_strategy filename]
    [-export_search_strategy filename] [-db database_name]
    [-dbsize num_letters] [-gilist filename] [-seqidlist filename]
    [-negative_gilist filename] [-negative_seqidlist filename]
    [-taxids taxids] [-negative_taxids taxids] [-taxidlist filename]
    [-negative_taxidlist filename] [-entrez_query entrez_query]
    [-subject subject_input_file] [-subject_loc range] [-query input_file]
    [-out output_file] [-evalue evalue] [-word_size int_value]
    [-gapopen open_penalty] [-gapextend extend_penalty]
    [-qcov_hsp_perc float_value] [-max_hsps int_value]
    [-xdrop_ungap float_value] [-xdrop_gap float_value]
    [-xdrop_gap_final float_value] [-searchsp int_value]
    [-sum_stats bool_value] [-seg SEG_options] [-soft_masking soft_masking]
    [-matrix matrix_name] [-threshold float_value] [-culling_limit int_value]
    [-best_hit_overhang float_value] [-best_hit_score_edge float_value]
    [-subject_besthit] [-window_size int_value] [-lcase_masking]
    [-query_loc range] [-parse_deflines] [-outfmt format] [-show_gis]
    [-num_descriptions int_value] [-num_alignments int_value]
    [-line_length line_length] [-html] [-max_target_seqs num_sequences]
    [-num_threads int_value] [-remote] [-comp_based_stats compo]
    [-use_sw_tback] [-gap_trigger float_value] [-num_iterations int_value]
    [-out_pssm checkpoint_file] [-out_ascii_pssm ascii_mtx_file]
    [-save_pssm_after_last_round] [-save_each_pssm] [-in_msa align_restart]
    [-msa_master_idx index] [-ignore_msa_master] [-in_pssm psi_chkpt_file]
    [-pseudocount pseudocount] [-inclusion_ethresh ethresh]
    [-phi_pattern file] [-version]

http://www.metagenomics.wiki/tools/blast/blastn-output-format-6

USAGE
  blastp [-h] [-help] [-import_search_strategy filename]
    [-export_search_strategy filename] [-task task_name] [-db database_name]
    [-dbsize num_letters] [-gilist filename] [-seqidlist filename]
    [-negative_gilist filename] [-negative_seqidlist filename]
    [-taxids taxids] [-negative_taxids taxids] [-taxidlist filename]
    [-negative_taxidlist filename] [-entrez_query entrez_query]
    [-db_soft_mask filtering_algorithm] [-db_hard_mask filtering_algorithm]
    [-subject subject_input_file] [-subject_loc range] [-query input_file]
    [-out output_file] [-evalue evalue] [-word_size int_value]
    [-gapopen open_penalty] [-gapextend extend_penalty]
    [-qcov_hsp_perc float_value] [-max_hsps int_value]
    [-xdrop_ungap float_value] [-xdrop_gap float_value]
    [-xdrop_gap_final float_value] [-searchsp int_value] [-seg SEG_options]
    [-soft_masking soft_masking] [-matrix matrix_name]
    [-threshold float_value] [-culling_limit int_value]
    [-best_hit_overhang float_value] [-best_hit_score_edge float_value]
    [-subject_besthit] [-window_size int_value] [-lcase_masking]
    [-query_loc range] [-parse_deflines] [-outfmt format] [-show_gis]
    [-num_descriptions int_value] [-num_alignments int_value]
    [-line_length line_length] [-html] [-max_target_seqs num_sequences]
    [-num_threads int_value] [-ungapped] [-remote] [-comp_based_stats compo]
    [-use_sw_tback] [-version]



x = readFASTA(system.file('protseq/P00750.fasta', package = 'Rcpi'))[[1]]
# }# NOT RUN {
dbpath = tempfile('tempdb', fileext = '.fasta')
invisible(file.copy(from = system.file('protseq/Plasminogen.fasta', package = 'Rcpi'), to = dbpath))
pssmmat = extractProtPSSM(seq = x, database.path = dbpath)
dim(pssmmat)  # 20 x 562 (P00750: length 562, 20 Amino Acids)
# }



## split seq 
. /u/local/Modules/default/init/modules.sh
module load python/3.7.2
cd /u/flashscratch/d/datduong/GOmultitask/blastp 
# python3 each_seq_to_fasta.py 
python3 read_pssm.py 

## make batch 
. /u/local/Modules/default/init/modules.sh
module load python/3.7.2
cd /u/flashscratch/d/datduong/GOmultitask/blastp 
# python3 each_seq_to_fasta.py 
python3 make_batch_pssm.py 
