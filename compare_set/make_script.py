import os,sys,re,pickle
import numpy as np 

## make life easier use python 

script = """

def_emb_dim='768'
metric_option='cosine' ## use cosine similarity as objective function 
nonlinear_gcnn='relu'

server='/local/datdb'
work_dir=$server/'goAndGeneAnnotationMar2017'
data_dir=$server/'goAndGeneAnnotationMar2017/entailment_data/AicScore/go_bert_cls'

result_folder=$work_dir/'BertGOName768temp/'
mkdir $result_folder
model_load=$result_folder/'best_state_dict.pytorch'

vector_file='/local/datdb/goAndGeneAnnotationMar2017/BertFineTuneGOEmb768/GOname_vector.pickle' ## use 2017 data here.
vocab_list='word_pubmed_intersect_GOdb.txt' # all vocab to be used, can be BERT or something, so we don't hardcode

conda activate tensorflow_gpuenv
cd $server/GOmultitask

pair=PAIR 
outDir=$server/'geneOrtholog'/$pair'Score/qnliFormatData17'

finalDir=$outDir/'BertGOName768'
mkdir $finalDir


for point in {0..ENDPOINT..GAP} ;  do 
  echo ' '
  echo 'iter '$point

  savePickle=$outDir/'GeneDict2test.'$point'.pickle'

  saveDf=$outDir/'Ortholog2testDef.'$point'.txt'
  test_file=$saveDf
  write_score=$finalDir/'Ortholog2testDef.'$point'.score.txt'

  ## set epoch=0 for testing
  CUDA_VISIBLE_DEVICES=6 python3 $server/GOmultitask/Precomputed_vector/encoder/do_model.py --main_dir $work_dir --qnli_dir $data_dir --batch_size_label 128 --result_folder $result_folder --epoch 0 --use_cuda --metric_option $metric_option --nonlinear_gcnn $nonlinear_gcnn --def_emb_dim $def_emb_dim --model_load $model_load --test_file $test_file --write_score $write_score --vocab_list $vocab_list --vector_file $vector_file > $result_folder/test2.log

  paste $test_file $write_score > $finalDir/'score.'$point'.txt' ## append columns 

  # rm -f $test_file
  rm -f $write_score

done

"""

os.chdir('/local/datdb/geneOrtholog')

# pair={'Human':[5300,50],'Yeast':[11700,300]}
pair={'HumanMouse':[22500,300],  
      'HumanFly':[10500,300], 
      'MouseFly':[12600,300], 
      'FlyWorm':[3600,300] }

fout = open("bertNameOrtho.sh","w")
for key,val in pair.items() : 
  script2 = re.sub("PAIR", "'"+key+"'", script)
  script2 = re.sub("ENDPOINT", str(val[0]) ,script2)
  script2 = re.sub("GAP", str(val[1]) ,script2)
  fout.write(script2+ "\n\n\n")


fout.close() 


