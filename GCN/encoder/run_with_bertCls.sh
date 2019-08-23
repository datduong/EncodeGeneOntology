


## use cosine similarity as objective function 
def_emb_dim='300'
gcn_native_emb='300'
metric_option='cosine'
nonlinear_gcnn='relu'

server='/local/datdb'
work_dir=$server/'deepgo/data'

data_where_train_is='goAndGeneAnnotationDec2018' ## where the train.csv is 
data_dir=$server/$data_where_train_is'/entailment_data/AicScore/go_bert_cls' ## whe

mkdir $work_dir/'GCNBertCls'
result_folder=$work_dir/'GCNBertCls/'$metric_option'.300' #$def_emb_dim.'clsVec'
mkdir $result_folder

word_mode='PretrainedGO'
w2v_emb='/local/datdb/deepgo/data/cosine.768.reduce300ClsVec/label_vector.pickle'

conda activate tensorflow_gpuenv
cd $server/GOmultitask

CUDA_VISIBLE_DEVICES=7 python3 $server/GOmultitask/GCN/encoder/do_model.py --w2v_emb $w2v_emb --lr 0.0001 --main_dir $work_dir --qnli_dir $data_dir --batch_size_label_desc 24 --batch_size_label 24 --result_folder $result_folder --epoch 100 --use_cuda --metric_option $metric_option --nonlinear_gcnn $nonlinear_gcnn --def_emb_dim $def_emb_dim --fix_word_emb --word_mode $word_mode > $result_folder/train.log




### **** use 2017 data so that later we can compare orthologs

## use cosine similarity as objective function 
def_emb_dim='300'
gcn_native_emb='300'
metric_option='cosine'
nonlinear_gcnn='relu'

server='/local/datdb'
work_dir=$server/'goAndGeneAnnotationMar2017' ## where the go.obo is at

data_where_train_is='goAndGeneAnnotationDec2018' ## where the train.csv is 
data_dir=$server/$data_where_train_is'/entailment_data/AicScore/go_bert_cls' ## whe

mkdir $work_dir/'GCNBertCls'
result_folder=$work_dir/'GCNBertCls/'$metric_option'.300' #$def_emb_dim.'clsVec'
mkdir $result_folder

word_mode='PretrainedGO'
w2v_emb='/local/datdb/deepgo/data/cosine.768.reduce300ClsVec/label_vector.pickle'

conda activate tensorflow_gpuenv
cd $server/GOmultitask

CUDA_VISIBLE_DEVICES=6 python3 $server/GOmultitask/GCN/encoder/do_model.py --w2v_emb $w2v_emb --lr 0.0001 --main_dir $work_dir --qnli_dir $data_dir --batch_size_label_desc 24 --batch_size_label 24 --result_folder $result_folder --epoch 100 --use_cuda --metric_option $metric_option --nonlinear_gcnn $nonlinear_gcnn --def_emb_dim $def_emb_dim --fix_word_emb --word_mode $word_mode > $result_folder/train.log

