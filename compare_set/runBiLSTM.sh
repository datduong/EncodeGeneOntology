
server='/local/datdb'

## use cosine similarity as objective function 
def_emb_dim='300'
metric_option='cosine'

work_dir=$server/'goAndGeneAnnotationMar2017'
data_dir=$server/'goAndGeneAnnotationMar2017/entailment_data/AicScore/go_bert_cls'


result_folder=$work_dir/$metric_option'.bilstm.300Vec'

model_load=$result_folder/'best_state_dict.pytorch'

conda activate tensorflow_gpuenv
cd $server/GOmultitask

pair='MouseFly' # 'MouseFly'
outDir=$server/'geneOrtholog'/$pair'Score/qnliFormatData17'
mkdir $outDir

finalDir=$outDir/$metric_option'.bilstm.300Vec'
mkdir $finalDir

w2v_emb=$work_dir'/word_pubmed_intersect_GOdb_w2v_pretrained.pickle'
vocab_list='word_pubmed_intersect_GOdb.txt' # all vocab to be used, can be BERT or something, so we don't hardcode 


for point in {0..12600..300} # 22500 3600
do 

echo ' '
echo 'iter '$point

savePickle=$outDir/'GeneDict2test.'$point'.pickle'

saveDf=$outDir/'Ortholog2testDef.'$point'.txt'
test_file=$saveDf
write_score=$finalDir/'Ortholog2testDef.'$point'.score.txt'


## set epoch=0 for testing

CUDA_VISIBLE_DEVICES=5 python3 $server/GOmultitask/biLSTM/encoder/do_model.py --model_load $model_load --fix_word_emb --vocab_list $vocab_list --w2v_emb $w2v_emb --lr 0.0001 --bilstm_dim 1024 --main_dir $work_dir --qnli_dir $data_dir --batch_size_label 96 --result_folder $result_folder --epoch 0 --use_cuda --metric_option $metric_option --def_emb_dim $def_emb_dim --reduce_cls_vec --test_file $test_file --write_score $write_score > $result_folder/test1.log


paste $test_file $write_score > $finalDir/'score.'$point'.txt' ## append columns 

# rm -f $test_file
rm -f $write_score

done


## 


#### ******




server='/local/datdb'

## use cosine similarity as objective function 
def_emb_dim='300'
metric_option='cosine'

work_dir=$server/'goAndGeneAnnotationMar2017'
data_dir=$server/'goAndGeneAnnotationMar2017/entailment_data/AicScore/go_bert_cls'


result_folder=$work_dir/$metric_option'.bilstm.300Vec'

model_load=$result_folder/'best_state_dict.pytorch'

conda activate tensorflow_gpuenv
cd $server/GOmultitask

pair='Human'
outDir=$server/$pair'PPI3ontology/qnliFormatData17'
mkdir $outDir

finalDir=$outDir/$metric_option'.bilstm.300Vec'
mkdir $finalDir

w2v_emb=$work_dir'/word_pubmed_intersect_GOdb_w2v_pretrained.pickle'
vocab_list='word_pubmed_intersect_GOdb.txt' # all vocab to be used, can be BERT or something, so we don't hardcode 


for point in {0..5300..50} # 22500 3600 11700 (5300,50)
do 

echo ' '
echo 'iter '$point

savePickle=$outDir/'GeneDict2test.'$point'.pickle'

saveDf=$outDir/'PPI2testDef.'$point'.txt'
test_file=$saveDf
write_score=$finalDir/'PPI2testDef.'$point'.score.txt'


## set epoch=0 for testing

CUDA_VISIBLE_DEVICES=7 python3 $server/GOmultitask/biLSTM/encoder/do_model.py --model_load $model_load --fix_word_emb --vocab_list $vocab_list --w2v_emb $w2v_emb --lr 0.0001 --bilstm_dim 1024 --main_dir $work_dir --qnli_dir $data_dir --batch_size_label 128 --result_folder $result_folder --epoch 0 --use_cuda --metric_option $metric_option --def_emb_dim $def_emb_dim --reduce_cls_vec --test_file $test_file --write_score $write_score > $result_folder/test1.log


paste $test_file $write_score > $finalDir/'score.'$point'.txt' ## append columns 

# rm -f $test_file
rm -f $write_score

done


## 



