


import pandas as pd 
import os,sys,re

main_script = '''cd /u/flashscratch/d/datduong/goAndGeneAnnotationDec2018/ecoli_yeast_human/full/redo_dec17_w2v17
/u/flashscratch/d/datduong/ncbi-blast-2.8.1+/bin/psiblast -db train_seq_has_rare_go.fasta -evalue 0.01 -query per_seq_fasta/GENE.fasta -out_ascii_pssm per_seq_pssm/GENE.pssm -out per_seq_out/GENE.out -num_iterations 3 -num_threads 4 '''


os.chdir ("/u/flashscratch/d/datduong/goAndGeneAnnotationDec2018/ecoli_yeast_human/full/redo_dec17_w2v17")
# file_name = '/u/flashscratch/d/datduong/goAndGeneAnnotationDec2018/ecoli_yeast_human/ecoli_yeast_human_seq_go_full_is_a_match_species.txt'
file_name = '/u/flashscratch/d/datduong/goAndGeneAnnotationDec2018/benchmark_cofactor_method/1079.testing/seq_tab_format.txt'
out_folder = 'per_seq_fasta'

df = pd.read_csv(file_name,dtype=str,sep="\t") ## "ArabidopsisUniprot.tab"
df = df.dropna()

gene_names = list (df['Entry'])
counter = 1
for g in gene_names: 
  script = re.sub("GENE",g,main_script)
  fout = open ("/u/flashscratch/d/datduong/temp/"+"job"+str(counter)+".sh","w")
  fout.write(script)
  fout.close() 
  counter = counter + 1 


import math 
output = "/u/flashscratch/d/datduong/temp/"
size = 100
counter = int (math.ceil(counter*1.0/size)*size )

script = '#!/bin/bash\n#$ -cwd\n#$ -o /u/flashscratch/d/datduong/test.$JOB_ID.out\n#$ -j y\n#$ -t 1-UPPER:SIZE\nfor i in `seq 0 ADDON`; do\n\tmy_task_id=$((SGE_TASK_ID + i))\n\toutputjob$my_task_id.sh\ndone\n'

script = re.sub('output',output,script) 
script = re.sub('UPPER',str(counter),script)
script = re.sub('SIZE',str(size),script)
script = re.sub('ADDON',str(size-1),script) 
	
fout = open (output+"submitJobs.sh","w")
fout.write(script)
fout.close()

os.system ("chmod 777 -R "+output)

# os.system ("qsub -l h_data=4G,highp,h_rt=12:50:50 -pe shared 4 " + output+"submitJobs.sh" ) 



 
