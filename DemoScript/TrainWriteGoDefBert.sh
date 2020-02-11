

#### train and write go def into vectors

##!! you will need to change folder paths

def_emb_dim='768' ## final dim of GO vectors
metric_option='cosine' ## we only support cosine similarity right now, but I guess there are other metric types.
server='/local/datdb'

#### you can download our data from google drive.

work_dir=$server/'goAndGeneAnnotationMar2017' ## main working dir
bert_model=$work_dir/'BERT_base_cased_tune_go_branch/fine_tune_lm_bioBERT' # use mask + next_sentence pre-trained model to innit. this is from Phase1 training

qnli_dir=$server/'goAndGeneAnnotationMar2017/entailment_data/AicScore/go_bert_cls' ## qnli for next-sentence prediction style ... needed only for Phase1
pregenerated_data=$server/'goAndGeneAnnotationMar2017/BERT_base_cased_tune_go_branch' # probably don't need this. we would use it if we do joint training with Phase1
bert_output_dir=$pregenerated_data/ExampleOutput
result_folder=$bert_output_dir ## should clean up redundant variable in source code.

conda activate tensorflow_gpuenv ## probably do not need tensorflow

cd $server/GOmultitask

#### train model

## --average_layer --layer_index -1 will take average of layer 12 output

CUDA_VISIBLE_DEVICES=0 python3 $server/GOmultitask/BERT/encoder/do_model.py --main_dir $work_dir --qnli_dir $qnli_dir --batch_size_aa_go 6 --batch_size_pretrain_bert 6 --bert_model $bert_model --pregenerated_data $pregenerated_data --bert_output_dir $bert_output_dir --result_folder $result_folder --epoch 1 --num_train_epochs_entailment 50 --use_cuda --metric_option $metric_option --def_emb_dim $def_emb_dim --reduce_cls_vec --average_layer --layer_index -1 > $result_folder/train.log


#### use below to write out GO vectors from a trained model

label_desc_dir=$work_dir/'go_def_in_obo.tsv' ## download from google drive or from uniprot
model_load=$result_folder/'best_state_dict.pytorch'

CUDA_VISIBLE_DEVICES=0 python3 $server/GOmultitask/BERT/encoder/do_model.py --main_dir $work_dir --qnli_dir $qnli_dir --batch_size_aa_go 8 --batch_size_pretrain_bert 8 --bert_model $bert_model --pregenerated_data $pregenerated_data --bert_output_dir $bert_output_dir --result_folder $result_folder --epoch 0 --num_train_epochs_entailment 0 --use_cuda --metric_option $metric_option --def_emb_dim $def_emb_dim --reduce_cls_vec --model_load $model_load --write_vector --label_desc_dir $label_desc_dir --average_layer --layer_index -1 > $result_folder/writevec.log


