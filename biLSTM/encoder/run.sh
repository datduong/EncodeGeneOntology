
## **** USE 2017 DATA
## use ENTAILMENT MODEL similarity as objective function 
def_emb_dim='300'
metric_option='entailment'
server='/local/datdb'

work_dir=$server/'goAndGeneAnnotationMar2017'
data_dir=$server/'goAndGeneAnnotationMar2017/entailment_data/AicScore/go_bert_cls'

w2v_emb=$work_dir'/word_pubmed_intersect_GOdb_w2v_pretrained.pickle'
vocab_list='word_pubmed_intersect_GOdb.txt' # all vocab to be used, can be BERT or something, so we don't hardcode 


result_folder=$work_dir/$metric_option'.bilstm.300Vec' #
mkdir $result_folder

conda activate tensorflow_gpuenv
cd $server/GOmultitask

CUDA_VISIBLE_DEVICES=5 python3 $server/GOmultitask/biLSTM/encoder/do_model.py --fix_word_emb --vocab_list $vocab_list --w2v_emb $w2v_emb --lr 0.0001 --bilstm_dim 1024 --main_dir $work_dir --qnli_dir $data_dir --batch_size_label 32 --result_folder $result_folder --epoch 200 --num_train_epochs_entailment 25 --use_cuda --metric_option $metric_option --def_emb_dim $def_emb_dim --reduce_cls_vec > $result_folder/train.log



## **** USE 2017 DATA ... but use 768-->768 
## use COSINE MODEL similarity as objective function 
def_emb_dim='768'
metric_option='cosine'
server='/local/datdb'

work_dir=$server/'goAndGeneAnnotationMar2017'
data_dir=$server/'goAndGeneAnnotationMar2017/entailment_data/AicScore/go_bert_cls'

w2v_emb=$work_dir'/word_pubmed_intersect_GOdb_w2v_pretrained.pickle'
vocab_list='word_pubmed_intersect_GOdb.txt' # all vocab to be used, can be BERT or something, so we don't hardcode 

result_folder=$work_dir/$metric_option'.bilstm'$def_emb_dim #
mkdir $result_folder

conda activate tensorflow_gpuenv
cd $server/GOmultitask

CUDA_VISIBLE_DEVICES=6 python3 $server/GOmultitask/biLSTM/encoder/do_model.py --fix_word_emb --vocab_list $vocab_list --w2v_emb $w2v_emb --lr 0.0001 --bilstm_dim 1024 --main_dir $work_dir --qnli_dir $data_dir --batch_size_label 32 --result_folder $result_folder --epoch 100 --use_cuda --metric_option $metric_option --def_emb_dim $def_emb_dim --reduce_cls_vec > $result_folder/train.log



## run on deepgo data 
## use COSINE MODEL similarity as objective function 
def_emb_dim='768'
metric_option='cosine'
server='/local/datdb'

work_dir=$server/'deepgo/data'
data_dir=$server/'goAndGeneAnnotationMar2017/entailment_data/AicScore/go_bert_cls'

w2v_emb=$work_dir'/word_pubmed_intersect_GOdb_w2v_pretrained.pickle'
vocab_list='word_pubmed_intersect_GOdb.txt' # all vocab to be used, can be BERT or something, so we don't hardcode 

result_folder=$work_dir/$metric_option'.bilstm'$def_emb_dim #
mkdir $result_folder

conda activate tensorflow_gpuenv
cd $server/GOmultitask

label_desc_dir=$work_dir/'go_def_in_obo.tsv'
model_load=$result_folder/'best_state_dict.pytorch'

CUDA_VISIBLE_DEVICES=5 python3 $server/GOmultitask/biLSTM/encoder/do_model.py --fix_word_emb --vocab_list $vocab_list --w2v_emb $w2v_emb --lr 0.0001 --bilstm_dim 1024 --main_dir $work_dir --qnli_dir $data_dir --batch_size_label 32 --result_folder $result_folder --epoch 100 --use_cuda --metric_option $metric_option --def_emb_dim $def_emb_dim --reduce_cls_vec > $result_folder/train.log

# CUDA_VISIBLE_DEVICES=5 python3 $server/GOmultitask/biLSTM/encoder/do_model.py --fix_word_emb --vocab_list $vocab_list --w2v_emb $w2v_emb --lr 0.0001 --bilstm_dim 1024 --main_dir $work_dir --qnli_dir $data_dir --batch_size_label 32 --result_folder $result_folder --epoch 0 --num_train_epochs_entailment 0 --use_cuda --metric_option $metric_option --def_emb_dim $def_emb_dim --reduce_cls_vec --model_load $model_load --write_vector --label_desc_dir $label_desc_dir > $result_folder/train2.log

