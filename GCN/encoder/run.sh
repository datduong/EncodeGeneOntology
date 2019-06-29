
## use cosine similarity as objective function 
def_emb_dim='300'
metric_option='cosine'
nonlinear_gcnn='relu'

server='/local/datdb'
work_dir=$server/'goAndGeneAnnotationDec2018'
bert_model=$work_dir/'BERT_base_cased_tune_go_branch/fine_tune_lm_bioBERT' # use the full mask + nextSentence to innit
data_dir=$server/'goAndGeneAnnotationDec2018/entailment_data/AicScore/go_bert_cls'
pregenerated_data=$server/'goAndGeneAnnotationDec2018/BERT_base_cased_tune_go_branch' # use the data of full mask + nextSentence to innit
bert_output_dir=$pregenerated_data/'fine_tune_lm_bioBERT'

mkdir $work_dir/'GCN'

result_folder=$work_dir/'GCN/'$metric_option'.300' #$def_emb_dim.'clsVec'
mkdir $result_folder

conda activate tensorflow_gpuenv
cd $server/GOmultitask

CUDA_VISIBLE_DEVICES=3 python3 $server/GOmultitask/GCN/encoder/do_model.py --lr 0.001 --main_dir $work_dir --qnli_dir $data_dir --batch_size_label 16 --batch_size_bert 8 --bert_model $bert_model --pregenerated_data $pregenerated_data --bert_output_dir $bert_output_dir --result_folder $result_folder --epoch 100 --use_cuda --metric_option $metric_option --nonlinear_gcnn $nonlinear_gcnn --def_emb_dim $def_emb_dim > $result_folder/train3.log


## **** do biLSTM

## use cosine similarity as objective function 
word_mode='bilstm'

def_emb_dim='300'
metric_option='cosine'
nonlinear_gcnn='relu'

server='/local/datdb'
work_dir=$server/'goAndGeneAnnotationDec2018'
bert_model=$work_dir/'BERT_base_cased_tune_go_branch/fine_tune_lm_bioBERT' # use the full mask + nextSentence to innit
data_dir=$server/'goAndGeneAnnotationDec2018/entailment_data/AicScore/go_bert_cls'
pregenerated_data=$server/'goAndGeneAnnotationDec2018/BERT_base_cased_tune_go_branch' # use the data of full mask + nextSentence to innit
bert_output_dir=$pregenerated_data/'fine_tune_lm_bioBERT'


w2v_emb=$work_dir'/word_pubmed_intersect_GOdb_w2v_pretrained.pickle'
vocab_list='word_pubmed_intersect_GOdb.txt' # all vocab to be used, can be BERT or something, so we don't hardcode 

mkdir $work_dir/'GCN'

result_folder=$work_dir/'GCN/bilstmEncMetric'$metric_option'300' #$def_emb_dim.'clsVec'
mkdir $result_folder

# model_load=$result_folder/'best_state_dict.pytorch'

conda activate tensorflow_gpuenv
cd $server/GOmultitask

CUDA_VISIBLE_DEVICES=4 python3 $server/GOmultitask/GCN/encoder/do_model.py --vocab_list $vocab_list --w2v_emb $w2v_emb --lr 0.001 --main_dir $work_dir --qnli_dir $data_dir --batch_size_label_desc 128 --batch_size_label 20 --batch_size_bert 8 --bert_model $bert_model --pregenerated_data $pregenerated_data --bert_output_dir $bert_output_dir --result_folder $result_folder --epoch 100 --use_cuda --metric_option $metric_option --nonlinear_gcnn $nonlinear_gcnn --def_emb_dim $def_emb_dim --fix_word_emb --word_mode $word_mode > $result_folder/train.log


## **** do cnn on words
word_mode='conv1d'

def_emb_dim='300'
metric_option='cosine'
nonlinear_gcnn='relu'

server='/local/datdb'
work_dir=$server/'goAndGeneAnnotationDec2018'
bert_model=$work_dir/'BERT_base_cased_tune_go_branch/fine_tune_lm_bioBERT' # use the full mask + nextSentence to innit
data_dir=$server/'goAndGeneAnnotationDec2018/entailment_data/AicScore/go_bert_cls'
pregenerated_data=$server/'goAndGeneAnnotationDec2018/BERT_base_cased_tune_go_branch' # use the data of full mask + nextSentence to innit
bert_output_dir=$pregenerated_data/'fine_tune_lm_bioBERT'


w2v_emb=$work_dir'/word_pubmed_intersect_GOdb_w2v_pretrained.pickle'
vocab_list='word_pubmed_intersect_GOdb.txt' # all vocab to be used, can be BERT or something, so we don't hardcode 

mkdir $work_dir/'GCN'

result_folder=$work_dir/'GCN/ConvEncMetric'$metric_option'300' #$def_emb_dim.'clsVec'
mkdir $result_folder

# model_load=$result_folder/'best_state_dict.pytorch'

conda activate tensorflow_gpuenv
cd $server/GOmultitask

CUDA_VISIBLE_DEVICES=3 python3 $server/GOmultitask/GCN/encoder/do_model.py --vocab_list $vocab_list --w2v_emb $w2v_emb --lr 0.001 --main_dir $work_dir --qnli_dir $data_dir --batch_size_label_desc 64 --batch_size_label 32 --bert_model $bert_model --pregenerated_data $pregenerated_data --bert_output_dir $bert_output_dir --result_folder $result_folder --epoch 100 --use_cuda --metric_option $metric_option --nonlinear_gcnn $nonlinear_gcnn --def_emb_dim $def_emb_dim --fix_word_emb --word_mode $word_mode > $result_folder/train.log



## **** do average pooling on words
word_mode='avepool'

def_emb_dim='300'
metric_option='cosine'
nonlinear_gcnn='relu'

server='/local/datdb'
work_dir=$server/'goAndGeneAnnotationDec2018'
bert_model=$work_dir/'BERT_base_cased_tune_go_branch/fine_tune_lm_bioBERT' # use the full mask + nextSentence to innit
data_dir=$server/'goAndGeneAnnotationDec2018/entailment_data/AicScore/go_bert_cls'
pregenerated_data=$server/'goAndGeneAnnotationDec2018/BERT_base_cased_tune_go_branch' # use the data of full mask + nextSentence to innit
bert_output_dir=$pregenerated_data/'fine_tune_lm_bioBERT'


w2v_emb=$work_dir'/word_pubmed_intersect_GOdb_w2v_pretrained.pickle'
vocab_list='word_pubmed_intersect_GOdb.txt' # all vocab to be used, can be BERT or something, so we don't hardcode 

mkdir $work_dir/'GCN'

result_folder=$work_dir/'GCN/AveEncMetric'$metric_option'300' #$def_emb_dim.'clsVec'
mkdir $result_folder

# model_load=$result_folder/'best_state_dict.pytorch'

conda activate tensorflow_gpuenv
cd $server/GOmultitask

CUDA_VISIBLE_DEVICES=4 python3 $server/GOmultitask/GCN/encoder/do_model.py --vocab_list $vocab_list --w2v_emb $w2v_emb --lr 0.001 --main_dir $work_dir --qnli_dir $data_dir --batch_size_label_desc 64 --batch_size_label 32 --bert_model $bert_model --pregenerated_data $pregenerated_data --bert_output_dir $bert_output_dir --result_folder $result_folder --epoch 100 --use_cuda --metric_option $metric_option --nonlinear_gcnn $nonlinear_gcnn --def_emb_dim $def_emb_dim --fix_word_emb --word_mode $word_mode > $result_folder/train.log

