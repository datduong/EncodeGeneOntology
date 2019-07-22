
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
