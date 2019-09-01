
## make data
module load python/3.7.2
work_dir='/u/flashscratch/d/datduong/goAndGeneAnnotationMar2017'
data_dir='/u/flashscratch/d/datduong/goAndGeneAnnotationMar2017/entailment_data/AicScore/go_bert_cls'

cd /u/flashscratch/d/datduong/GOmultitask/BERT/entailment
python3 do_model.py --main_dir $work_dir --qnli_dir $data_dir --batch_size_label 32


## **** USE 2017 DATA
## use cosine similarity as objective function 
def_emb_dim='300'
metric_option='cosine'
server='/local/datdb'

work_dir=$server/'goAndGeneAnnotationMar2017'
bert_model=$work_dir/'BERT_base_cased_tune_go_branch/fine_tune_lm_bioBERT' # use the full mask + nextSentence to innit
data_dir=$server/'goAndGeneAnnotationMar2017/entailment_data/AicScore/go_bert_cls'
pregenerated_data=$server/'goAndGeneAnnotationMar2017/BERT_base_cased_tune_go_branch' # use the data of full mask + nextSentence to innit
bert_output_dir=$pregenerated_data/'fine_tune_lm_bioBERT'
mkdir $bert_output_dir

result_folder=$bert_output_dir/$metric_option'.768.300.Joint2ndLastWithCls' #$def_emb_dim.'clsVec'
mkdir $result_folder

conda activate tensorflow_gpuenv
cd $server/GOmultitask
CUDA_VISIBLE_DEVICES=7 python3 $server/GOmultitask/BERT/encoder/do_model.py --main_dir $work_dir --qnli_dir $data_dir --batch_size_label 8 --batch_size_bert 8 --bert_model $bert_model --pregenerated_data $pregenerated_data --bert_output_dir $bert_output_dir --result_folder $result_folder --epoch 3 --num_train_epochs_entailment 10 --num_train_epochs_bert 2 --use_cuda --metric_option $metric_option --def_emb_dim $def_emb_dim --reduce_cls_vec --update_bert --average_layer > $result_folder/train.log
## use below to write out GO vectors
label_desc_dir=$work_dir/'go_def_in_obo.tsv'
model_load=$result_folder/'best_state_dict.pytorch'
CUDA_VISIBLE_DEVICES=7 python3 $server/GOmultitask/BERT/encoder/do_model.py --main_dir $work_dir --qnli_dir $data_dir --batch_size_label 8 --batch_size_bert 8 --bert_model $bert_model --pregenerated_data $pregenerated_data --bert_output_dir $bert_output_dir --result_folder $result_folder --epoch 0 --num_train_epochs_entailment 0 --use_cuda --metric_option $metric_option --def_emb_dim $def_emb_dim --reduce_cls_vec --model_load $model_load --write_vector --label_desc_dir $label_desc_dir --average_layer > $result_folder/writevec.log



########## try 786-->786 


## **** USE 2017 DATA
## use cosine similarity as objective function 
def_emb_dim='768'
metric_option='cosine'
server='/local/datdb'

work_dir=$server/'goAndGeneAnnotationMar2017'
bert_model=$work_dir/'BERT_base_cased_tune_go_branch/fine_tune_lm_bioBERT' # use the full mask + nextSentence to innit
data_dir=$server/'goAndGeneAnnotationMar2017/entailment_data/AicScore/go_bert_cls'
pregenerated_data=$server/'goAndGeneAnnotationMar2017/BERT_base_cased_tune_go_branch' # use the data of full mask + nextSentence to innit
bert_output_dir=$pregenerated_data/'fine_tune_lm_bioBERT'
mkdir $bert_output_dir

result_folder=$bert_output_dir/$metric_option'.Cls768.Linear768' #$def_emb_dim.'clsVec'
mkdir $result_folder

conda activate tensorflow_gpuenv
cd $server/GOmultitask
CUDA_VISIBLE_DEVICES=7 python3 $server/GOmultitask/BERT/encoder/do_model.py --main_dir $work_dir --qnli_dir $data_dir --batch_size_label 8 --batch_size_bert 8 --bert_model $bert_model --pregenerated_data $pregenerated_data --bert_output_dir $bert_output_dir --result_folder $result_folder --epoch 1 --num_train_epochs_entailment 100 --use_cuda --metric_option $metric_option --def_emb_dim $def_emb_dim --reduce_cls_vec  > $result_folder/train.log
## use below to write out GO vectors
label_desc_dir=$work_dir/'go_def_in_obo.tsv'
model_load=$result_folder/'best_state_dict.pytorch'
CUDA_VISIBLE_DEVICES=7 python3 $server/GOmultitask/BERT/encoder/do_model.py --main_dir $work_dir --qnli_dir $data_dir --batch_size_label 8 --batch_size_bert 8 --bert_model $bert_model --pregenerated_data $pregenerated_data --bert_output_dir $bert_output_dir --result_folder $result_folder --epoch 0 --num_train_epochs_entailment 0 --use_cuda --metric_option $metric_option --def_emb_dim $def_emb_dim --reduce_cls_vec --model_load $model_load --write_vector --label_desc_dir $label_desc_dir > $result_folder/writevec.log




## **** USE DEEPGO DATA
def_emb_dim='768'
metric_option='cosine' ## use cosine similarity as objective function 
server='/local/datdb'

work_dir=$server/'goAndGeneAnnotationMar2017' ## defintions will not change very much, if we can capture words relationship well, it should be okay
bert_model=$work_dir/'BERT_base_cased_tune_go_branch/fine_tune_lm_bioBERT' # use the full mask + nextSentence to innit
data_dir=$server/'goAndGeneAnnotationMar2017/entailment_data/AicScore/go_bert_cls'
pregenerated_data=$server/'goAndGeneAnnotationMar2017/BERT_base_cased_tune_go_branch' # use the data of full mask + nextSentence to innit

work_dir=$server/'deepgo/data' ## redefine
mkdir $bert_output_dir
result_folder=$work_dir/$metric_option'.Cls768.Linear768.Layer12'
mkdir $result_folder

conda activate tensorflow_gpuenv
cd $server/GOmultitask
CUDA_VISIBLE_DEVICES=7 python3 $server/GOmultitask/BERT/encoder/do_model.py --main_dir $work_dir --qnli_dir $data_dir --batch_size_label 8 --batch_size_bert 8 --bert_model $bert_model --pregenerated_data $pregenerated_data --bert_output_dir $bert_output_dir --result_folder $result_folder --epoch 1 --num_train_epochs_entailment 100 --use_cuda --metric_option $metric_option --def_emb_dim $def_emb_dim --reduce_cls_vec --average_layer --layer_index 1 > $result_folder/train.log
## use below to write out GO vectors
label_desc_dir=$work_dir/'go_def_in_obo.tsv'
model_load=$result_folder/'best_state_dict.pytorch'
CUDA_VISIBLE_DEVICES=7 python3 $server/GOmultitask/BERT/encoder/do_model.py --main_dir $work_dir --qnli_dir $data_dir --batch_size_label 8 --batch_size_bert 8 --bert_model $bert_model --pregenerated_data $pregenerated_data --bert_output_dir $bert_output_dir --result_folder $result_folder --epoch 0 --num_train_epochs_entailment 0 --use_cuda --metric_option $metric_option --def_emb_dim $def_emb_dim --reduce_cls_vec --model_load $model_load --write_vector --label_desc_dir $label_desc_dir --average_layer --layer_index 1 > $result_folder/writevec.log


