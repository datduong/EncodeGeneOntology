# note: using bilstm_dim 1024 but def_emb_dim 300, changed to 100 epoch instead of 200

cuda_device='1'
date
## **** USE 2017 DATA
## use ENTAILMENT MODEL similarity as objective function 
echo '\n\n##########################\n2017 entailment'
def_emb_dim='300'
metric_option='entailment'
server='/local/lgai'

work_dir=$server/'goAndGeneAnnotationMar2017'
data_dir=$server/'goAndGeneAnnotationMar2017/entailment_data/AicScore/go_bert_cls'

w2v_emb=$work_dir'/word_pubmed_intersect_GOdb_w2v_pretrained.pickle'
vocab_list=$work_dir'/word_pubmed_intersect_GOdb.txt' # all vocab to be used, can be BERT or something, so we don't hardcode 


result_folder=$work_dir/$metric_option'.bilstm.300Vec' #
mkdir $result_folder

#conda activate tensorflow_gpuenv
cd $server/EncodeGeneOntology

CUDA_VISIBLE_DEVICES=${cuda_device} python3 $server/EncodeGeneOntology/biLSTM/encoder/do_model.py --fix_word_emb --vocab_list $vocab_list --w2v_emb $w2v_emb --lr 0.0001 --bilstm_dim 1024 --main_dir $work_dir --qnli_dir $data_dir --batch_size_label 32 --result_folder $result_folder --epoch 100 --num_train_epochs_entailment 25 --use_cuda --metric_option $metric_option --def_emb_dim $def_emb_dim --reduce_cls_vec --write_vector > $result_folder/train.log 

echo '\n'; date


## **** USE 2017 DATA
## use COSINE MODEL similarity as objective function 
echo '\n\n##########################\n2017 cosine'

def_emb_dim='300'
metric_option='cosine'
server='/local/lgai'

work_dir=$server/'goAndGeneAnnotationMar2017'
data_dir=$server/'goAndGeneAnnotationMar2017/entailment_data/AicScore/go_bert_cls'

w2v_emb=$work_dir'/word_pubmed_intersect_GOdb_w2v_pretrained.pickle'
vocab_list=$work_dir'/word_pubmed_intersect_GOdb.txt' # all vocab to be used, can be BERT or something, so we don't hardcode 


result_folder=$work_dir/$metric_option'.bilstm.300Vec_2017' #
mkdir $result_folder

#conda activate tensorflow_gpuenv
cd $server/EncodeGeneOntology

CUDA_VISIBLE_DEVICES=${cuda_device} python3 $server/EncodeGeneOntology/biLSTM/encoder/do_model.py --fix_word_emb --vocab_list $vocab_list --w2v_emb $w2v_emb --lr 0.0001 --bilstm_dim 1024 --main_dir $work_dir --qnli_dir $data_dir --batch_size_label 32 --result_folder $result_folder --epoch 100 --num_train_epochs_entailment 25 --use_cuda --metric_option $metric_option --def_emb_dim $def_emb_dim --reduce_cls_vec --write_vector > $result_folder/train.log

echo '\n'; date

## run on deepgo data 
## use COSINE MODEL similarity as objective function 
echo '\n\n##########################\ndeepgo cosine'

def_emb_dim='300'
metric_option='cosine'
server='/local/lgai'

work_dir=$server/'deepgo/data'
data_dir=$server/'goAndGeneAnnotationMar2017/entailment_data/AicScore/go_bert_cls'

w2v_emb=$work_dir'/word_pubmed_intersect_GOdb_w2v_pretrained.pickle'
vocab_list=$work_dir'/word_pubmed_intersect_GOdb.txt' # all vocab to be used, can be BERT or something, so we don't hardcode 


result_folder=$work_dir/$metric_option'.bilstm.300Vec_deepgo' #
mkdir $result_folder

#conda activate tensorflow_gpuenv
cd $server/GOmultitask

CUDA_VISIBLE_DEVICES=${cuda_device} python3 $server/GOmultitask/biLSTM/encoder/do_model.py --fix_word_emb --vocab_list $vocab_list --w2v_emb $w2v_emb --lr 0.0001 --bilstm_dim 1024 --main_dir $work_dir --qnli_dir $data_dir --batch_size_label 32 --result_folder $result_folder --epoch 100 --num_train_epochs_entailment 25 --use_cuda --metric_option $metric_option --def_emb_dim $def_emb_dim --reduce_cls_vec --write_vector > $result_folder/train.log

echo '\n'; date