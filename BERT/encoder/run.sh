
## make data
module load python/3.7.2
work_dir='/u/flashscratch/d/datduong/goAndGeneAnnotationDec2018'
data_dir='/u/flashscratch/d/datduong/goAndGeneAnnotationDec2018/entailment_data/AicScore/go_bert_cls'

cd /u/flashscratch/d/datduong/GOmultitask/BERT/entailment
python3 do_model.py --main_dir $work_dir --qnli_dir $data_dir --batch_size_label 32


## use entailment model
def_emb_dim='300'
metric_option='entailment' ## able to do fp16 
server='/local/datdb'
work_dir=$server/'goAndGeneAnnotationDec2018'
bert_model=$work_dir/'BERT_base_cased_tune_go_branch/fine_tune_lm_bioBERT' # use the full mask + nextSentence to innit
data_dir=$server/'goAndGeneAnnotationDec2018/entailment_data/AicScore/go_bert_cls'
pregenerated_data=$server/'goAndGeneAnnotationDec2018/BERT_base_cased_tune_go_branch' # use the data of full mask + nextSentence to innit
bert_output_dir=$pregenerated_data/'fine_tune_lm_bioBERT'
mkdir $bert_output_dir

result_folder=$bert_output_dir/'joint.'$metric_option.$def_emb_dim
mkdir $result_folder

# model_load=$result_folder/'current_state_dict_inner_epoch1.pytorch'

conda activate tensorflow_gpuenv
cd $server/GOmultitask

CUDA_VISIBLE_DEVICES=2 python3 $server/GOmultitask/BERT/entailment/do_model.py --main_dir $work_dir --qnli_dir $data_dir --batch_size_label 12 --batch_size_bert 10 --bert_model $bert_model --pregenerated_data $pregenerated_data --bert_output_dir $bert_output_dir --result_folder $result_folder --epoch 1 --num_train_epochs_entailment 10 --num_train_epochs_bert 2 --use_cuda --metric_option $metric_option --fp16 --def_emb_dim $def_emb_dim > joint.$metric_option.$def_emb_dim.log

# --do_continue --model_load $model_load

## use cosine similarity as objective function 
def_emb_dim='300'
metric_option='cosine'
server='/local/datdb'

work_dir=$server/'goAndGeneAnnotationDec2018'
bert_model=$work_dir/'BERT_base_cased_tune_go_branch/fine_tune_lm_bioBERT' # use the full mask + nextSentence to innit
data_dir=$server/'goAndGeneAnnotationDec2018/entailment_data/AicScore/go_bert_cls'
pregenerated_data=$server/'goAndGeneAnnotationDec2018/BERT_base_cased_tune_go_branch' # use the data of full mask + nextSentence to innit
bert_output_dir=$pregenerated_data/'fine_tune_lm_bioBERT'
mkdir $bert_output_dir

result_folder=$bert_output_dir/$metric_option'.cosine.300.reduceClsVec' #$def_emb_dim.'clsVec'
mkdir $result_folder

conda activate tensorflow_gpuenv
cd $server/GOmultitask

CUDA_VISIBLE_DEVICES=2 python3 $server/GOmultitask/BERT/entailment/do_model.py --main_dir $work_dir --qnli_dir $data_dir --batch_size_label 8 --batch_size_bert 8 --bert_model $bert_model --pregenerated_data $pregenerated_data --bert_output_dir $bert_output_dir --result_folder $result_folder --epoch 1 --num_train_epochs_entailment 25 --num_train_epochs_bert 2 --use_cuda --metric_option $metric_option --def_emb_dim $def_emb_dim --reduce_cls_vec > $result_folder/train.log



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

result_folder=$bert_output_dir/$metric_option'.768.reduce300ClsVec' #$def_emb_dim.'clsVec'
mkdir $result_folder

conda activate tensorflow_gpuenv
cd $server/GOmultitask

CUDA_VISIBLE_DEVICES=7 python3 $server/GOmultitask/BERT/encoder/do_model.py --main_dir $work_dir --qnli_dir $data_dir --batch_size_label 8 --batch_size_bert 8 --bert_model $bert_model --pregenerated_data $pregenerated_data --bert_output_dir $bert_output_dir --result_folder $result_folder --epoch 1 --num_train_epochs_entailment 25 --num_train_epochs_bert 2 --use_cuda --metric_option $metric_option --def_emb_dim $def_emb_dim --reduce_cls_vec > $result_folder/train.log


## **** USE 2017 DATA
## use ENTAILMENT MODEL similarity as objective function 
def_emb_dim='300'
metric_option='entailment'
server='/local/datdb'

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

CUDA_VISIBLE_DEVICES=6 python3 $server/GOmultitask/BERT/encoder/do_model.py --main_dir $work_dir --qnli_dir $data_dir --batch_size_label 12 --batch_size_bert 8 --bert_model $bert_model --pregenerated_data $pregenerated_data --bert_output_dir $bert_output_dir --result_folder $result_folder --epoch 1 --num_train_epochs_entailment 25 --num_train_epochs_bert 2 --use_cuda --metric_option $metric_option --def_emb_dim $def_emb_dim --reduce_cls_vec --fp16 --model_load $model_load > $result_folder/train.log





## **** USE Deepgo DATA
## use cosine similarity as objective function 
def_emb_dim='300'
metric_option='cosine'
server='/local/datdb'

work_dir=$server/'deepgo/data'
data_dir=$server/'deepgo/data/entailment_data/AicScore/go_bert_cls'


bert_model=$server/'goAndGeneAnnotationDec2018/BERT_base_cased_tune_go_branch/fine_tune_lm_bioBERT' # use the full mask + nextSentence to innit
pregenerated_data=$server/'goAndGeneAnnotationDec2018/BERT_base_cased_tune_go_branch' # use the data of full mask + nextSentence to innit
bert_output_dir=$pregenerated_data/'fine_tune_lm_bioBERT'
# mkdir $bert_output_dir

result_folder=$work_dir/$metric_option'.768.reduce300ClsVec' #$def_emb_dim.'clsVec'
mkdir $result_folder

conda activate tensorflow_gpuenv
cd $server/GOmultitask

CUDA_VISIBLE_DEVICES=7 python3 $server/GOmultitask/BERT/encoder/do_model.py --main_dir $work_dir --qnli_dir $data_dir --batch_size_label 8 --batch_size_bert 8 --bert_model $bert_model --pregenerated_data $pregenerated_data --bert_output_dir $bert_output_dir --result_folder $result_folder --epoch 1 --num_train_epochs_entailment 25 --num_train_epochs_bert 2 --use_cuda --metric_option $metric_option --def_emb_dim $def_emb_dim --reduce_cls_vec > $result_folder/train.log


