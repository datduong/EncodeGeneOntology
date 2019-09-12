



### **** use 2017 data so that later we can compare orthologs

## use cosine similarity as objective function 
def_emb_dim='256'
gcnn_dim='256'
metric_option='cosine'
nonlinear_gcnn='relu'

server='/local/datdb'
work_dir=$server/'goAndGeneAnnotationMar2017' ## where the go.obo is at

data_where_train_is='goAndGeneAnnotationMar2017' ## where the train.csv is 
data_dir=$server/$data_where_train_is'/entailment_data/AicScore/go_bert_cls' ## whe
mkdir $work_dir/'GCNBertClsL12JointTrain'
result_folder=$work_dir/'GCNBertClsL12JointTrain/' # GCNBertAveClsSepL12JointTrain
mkdir $result_folder
model_load=$result_folder/'best_state_dict.pytorch'

## handle bert input 
bert_model=$work_dir/'BERT_base_cased_tune_go_branch/fine_tune_lm_bioBERT' # use the full mask + nextSentence to innit

conda activate tensorflow_gpuenv
cd $server/GOmultitask

CUDA_VISIBLE_DEVICES=1 python3 $server/GOmultitask/GCN/encoder/do_model_add_encoder.py --bert_model $bert_model --main_dir $work_dir --qnli_dir $data_dir --batch_size_label_desc 8 --batch_size_label 8 --result_folder $result_folder --epoch 100 --use_cuda --metric_option $metric_option --nonlinear_gcnn $nonlinear_gcnn --def_emb_dim $def_emb_dim --gcnn_dim $gcnn_dim --model_load $model_load > $result_folder/train.log


# --layer_index 1 --average_layer
# --reduce_cls_vec 


### **** use 2017 data so that later we can compare orthologs
### AVERAGE LAST LAYER

## use cosine similarity as objective function 
def_emb_dim='256'
gcnn_dim='256'
metric_option='cosine'
nonlinear_gcnn='relu'

server='/local/datdb'
work_dir=$server/'goAndGeneAnnotationMar2017' ## where the go.obo is at

data_where_train_is='goAndGeneAnnotationMar2017' ## where the train.csv is 
data_dir=$server/$data_where_train_is'/entailment_data/AicScore/go_bert_cls' ## whe
mkdir $work_dir/'GCNBertAveClsSepL12JointTrain'
result_folder=$work_dir/'GCNBertAveClsSepL12JointTrain/' # GCNBertAveClsSepL12JointTrain
mkdir $result_folder
model_load=$result_folder/'best_state_dict.pytorch'
## handle bert input 
bert_model=$work_dir/'BERT_base_cased_tune_go_branch/fine_tune_lm_bioBERT' # use the full mask + nextSentence to innit

conda activate tensorflow_gpuenv
cd $server/GOmultitask

CUDA_VISIBLE_DEVICES=3 python3 $server/GOmultitask/GCN/encoder/do_model_add_encoder.py --bert_model $bert_model --main_dir $work_dir --qnli_dir $data_dir --batch_size_label_desc 8 --batch_size_label 8 --result_folder $result_folder --epoch 100 --use_cuda --metric_option $metric_option --nonlinear_gcnn $nonlinear_gcnn --def_emb_dim $def_emb_dim --gcnn_dim $gcnn_dim --layer_index 1 --average_layer --model_load $model_load > $result_folder/train.log


