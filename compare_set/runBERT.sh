
server='/local/datdb'

## use cosine similarity as objective function 
def_emb_dim='768'
metric_option='cosine'

work_dir=$server/'goAndGeneAnnotationMar2017'
bert_model=$work_dir/'BERT_base_cased_tune_go_branch/fine_tune_lm_bioBERT' # use the full mask + nextSentence to innit
data_dir=$server/'goAndGeneAnnotationMar2017/entailment_data/AicScore/go_bert_cls'
pregenerated_data=$server/'goAndGeneAnnotationMar2017/BERT_base_cased_tune_go_branch' # use the data of full mask + nextSentence to innit
bert_output_dir=$pregenerated_data/'fine_tune_lm_bioBERT'

result_folder=$bert_output_dir/cosine.AveWordClsSep768.Linear768.Layer12
mkdir $result_folder

model_load=$result_folder/'best_state_dict.pytorch'

conda activate tensorflow_gpuenv
cd $server/GOmultitask

pair='HumanFly' # 'HumanMouse'
outDir=$server/'geneOrtholog'/$pair'Score/qnliFormatData17'
mkdir $outDir
finalDir=$outDir/cosine.AveWordClsSep768.Linear768.Layer12
mkdir $finalDir

for point in {0..10800..300} # 12600
do 

  echo ' '
  echo 'iter '$point

  savePickle=$outDir/'GeneDict2test.'$point'.pickle'
  saveDf=$outDir/'Ortholog2testDef.'$point'.txt'
  test_file=$saveDf
  write_score=$finalDir/'Ortholog2testDef.'$point'.score.txt'

  ## set epoch=0 for testing
  CUDA_VISIBLE_DEVICES=5 python3 $server/GOmultitask/BERT/encoder/do_model.py --main_dir $work_dir --qnli_dir $data_dir --batch_size_aa_go 128 --batch_size_pretrain_bert 8 --bert_model $bert_model --pregenerated_data $pregenerated_data --bert_output_dir $bert_output_dir --result_folder $result_folder --epoch 0 --num_train_epochs_entailment 0 --use_cuda --metric_option $metric_option --def_emb_dim $def_emb_dim --reduce_cls_vec --model_load $model_load --write_score $write_score --test_file $test_file --average_layer --layer_index 1 > $result_folder/test1.log

  paste $test_file $write_score > $finalDir/'score.'$point'.txt' ## append columns 
  # rm -f $test_file
  rm -f $write_score

done


## 



### **** COMPARE GENE-GENE INTERACTION WITHIN 1 SPECIES



server='/local/datdb'

## use cosine similarity as objective function 
def_emb_dim='300'
metric_option='cosine'

work_dir=$server/'goAndGeneAnnotationMar2017'
bert_model=$work_dir/'BERT_base_cased_tune_go_branch/fine_tune_lm_bioBERT' # use the full mask + nextSentence to innit
data_dir=$server/'goAndGeneAnnotationMar2017/entailment_data/AicScore/go_bert_cls'
pregenerated_data=$server/'goAndGeneAnnotationMar2017/BERT_base_cased_tune_go_branch' # use the data of full mask + nextSentence to innit
bert_output_dir=$pregenerated_data/'fine_tune_lm_bioBERT'
mkdir $bert_output_dir

result_folder=$bert_output_dir/$metric_option'.768.reduce300ClsVec' #$def_emb_dim.'clsVec'
mkdir $result_folder

model_load=$result_folder/'best_state_dict.pytorch'

conda activate tensorflow_gpuenv
cd $server/GOmultitask

pair='Human' # 'HumanMouse'
outDir=$server/$pair'PPI3ontology/qnliFormatData17'
mkdir $outDir

finalDir=$outDir/$metric_option'.768.reduce300ClsVec'
mkdir $finalDir


for point in {0..5300..50} # 12600 11700
do 

echo ' '
echo 'iter '$point

savePickle=$outDir/'GeneDict2test.'$point'.pickle'

saveDf=$outDir/'PPI2testDef.'$point'.txt'
test_file=$saveDf

write_score=$finalDir/'PPI2testDef.'$point'.score.txt'


## set epoch=0 for testing
CUDA_VISIBLE_DEVICES=6 python3 $server/GOmultitask/BERT/encoder/do_model.py --main_dir $work_dir --qnli_dir $data_dir --batch_size_aa_go 64 --batch_size_pretrain_bert 8 --bert_model $bert_model --pregenerated_data $pregenerated_data --bert_output_dir $bert_output_dir --result_folder $result_folder --epoch 0 --num_train_epochs_entailment 0 --use_cuda --metric_option $metric_option --def_emb_dim $def_emb_dim --reduce_cls_vec --model_load $model_load --write_score $write_score --test_file $test_file > $result_folder/test1.log

paste $test_file $write_score > $finalDir/'score.'$point'.txt' ## append columns 

# rm -f $test_file
rm -f $write_score

done







server='/local/datdb'

## use cosine similarity as objective function 
def_emb_dim='768'
metric_option='cosine'

work_dir=$server/'goAndGeneAnnotationMar2017'
bert_model=$work_dir/'BERT_base_cased_tune_go_branch/fine_tune_lm_bioBERT' # use the full mask + nextSentence to innit
data_dir=$server/'goAndGeneAnnotationMar2017/entailment_data/AicScore/go_bert_cls'
pregenerated_data=$server/'goAndGeneAnnotationMar2017/BERT_base_cased_tune_go_branch' # use the data of full mask + nextSentence to innit
bert_output_dir=$pregenerated_data/'fine_tune_lm_bioBERT'

result_folder=$bert_output_dir/cosine.Cls768.Linear768
mkdir $result_folder

model_load=$result_folder/'best_state_dict.pytorch'

conda activate tensorflow_gpuenv
cd $server/GOmultitask

pair=PAIR # 'HumanMouse'
outDir=$server/'geneOrtholog'/$pair'Score/qnliFormatData17'
mkdir $outDir
finalDir=$outDir/cosineCls768Linear768
mkdir $finalDir

for point in {0..ENDPOINT..GAP} # 12600
do 

  echo ' '
  echo 'iter '$point

  # savePickle=$outDir/'GeneDict2test.'$point'.pickle'
  saveDf=$outDir/'Ortholog2testDef.'$point'.txt'
  test_file=$saveDf
  write_score=$finalDir/'Ortholog2testDef.'$point'.score.txt'

  ## set epoch=0 for testing
  CUDA_VISIBLE_DEVICES=1 python3 $server/GOmultitask/BERT/encoder/do_model.py --main_dir $work_dir --qnli_dir $data_dir --batch_size_aa_go 64 --batch_size_pretrain_bert 8 --bert_model $bert_model --pregenerated_data $pregenerated_data --bert_output_dir $bert_output_dir --result_folder $result_folder --epoch 0 --num_train_epochs_entailment 0 --use_cuda --metric_option $metric_option --def_emb_dim $def_emb_dim --reduce_cls_vec --model_load $model_load --write_score $write_score --test_file $test_file > $result_folder/test1.log

  paste $test_file $write_score > $finalDir/'score.'$point'.txt' ## append columns 
  rm -f $write_score

done

