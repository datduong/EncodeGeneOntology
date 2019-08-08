#!/bin/sh

# python
#mypython3='/local/lgai/anaconda3/bin/python3.7'

# use cosine similarity as objective function 
word_mode='pretrained_aux' #'pretrained'

def_emb_dim='400' # input dimension to GCN (including any portion of embedding to be trained)
metric_option='cosine'
nonlinear_gcnn='relu'
pre_def_emb_dim='300' # how many dims are pretrained - used for file name, not arg
aux_def_emb_dim='100' # how many dims to be trained
n_epoch='10'

server='/local/lgai'
work_dir=$server/'goAndGeneAnnotationAug2019'
bert_model=$work_dir/'BERT_base_cased_tune_go_branch/fine_tune_lm_bioBERT' # use the full mask + nextSentence to innit
data_dir=$work_dir/'entailment_data/AicScore/go_bert_cls'
pregenerated_data=$server/'goAndGeneAnnotationDec2018/BERT_base_cased_tune_go_branch' # use the data of full mask + nextSentence to innit
bert_output_dir=$pregenerated_data/'fine_tune_lm_bioBERT'


w2v_emb='/local/datdb/goAndGeneAnnotationMar2017/BERT_base_cased_tune_go_branch/fine_tune_lm_bioBERT/cosine.768.reduce300ClsVec/label_vector.txt'
vocab_list='word_pubmed_intersect_GOdb.txt' # all vocab to be used, can be BERT or something, so we don't hardcode 

mkdir $work_dir/'GCN'

result_folder=$work_dir/'GCN/'$metric_option'_pre'$pre_def_emb_dim'_aux'$aux_def_emb_dim'_epoch'$n_epoch #$def_emb_dim.'clsVec'
mkdir $result_folder

# model_load=$result_folder/'best_state_dict.pytorch'

source activate tensorflow_gpuenv

# conda activate tensorflow_gpuenv

cd $server/EncodeGeneOntology

CUDA_VISIBLE_DEVICES=1 /local/lgai/anaconda3/bin/python3.7 $server/EncodeGeneOntology/GCN/encoder/do_model.py \
--vocab_list $vocab_list  --main_dir $work_dir --qnli_dir $data_dir   \
--lr 0.001 --epoch $n_epoch --use_cuda --metric_option $metric_option \
--batch_size_label_desc 128 --batch_size_label 20 --batch_size_bert 8 \
--bert_model $bert_model --pregenerated_data $pregenerated_data       \
--bert_output_dir $bert_output_dir                                    \
--result_folder $result_folder                                        \
--nonlinear_gcnn $nonlinear_gcnn                                      \
--w2v_emb $w2v_emb   --fix_word_emb                                   \
--def_emb_dim $def_emb_dim                                            \
 --aux_def_emb_dim $aux_def_emb_dim  \
--word_mode $word_mode > $result_folder/train.log
#--pre_def_emb_dim $pre_def_emb_dim  # currently not option

echo 'done'
date


