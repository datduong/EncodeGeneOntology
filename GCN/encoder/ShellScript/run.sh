
## use cosine similarity as objective function 
def_emb_dim='768'
gcnn_dim='768'
metric_option='cosine'
nonlinear_gcnn='relu'

data_where_train_is='goAndGeneAnnotationMar2017' ## where the train.csv is 
which_data='goAndGeneAnnotationMar2017' ## where the go.obo is 
server='/local/datdb'
work_dir=$server/$which_data
data_dir=$server/$data_where_train_is/'entailment_data/AicScore/go_bert_cls'
mkdir $work_dir/'GCN'

result_folder=$work_dir/'GCN/'$metric_option'.'$def_emb_dim #$def_emb_dim.'clsVec'
mkdir $result_folder

conda activate tensorflow_gpuenv
cd $server/GOmultitask

## notice we do not add the reduce-flag to call the linear layer in cosine-module. because this would not be the original implementation of GCN
## NOT USE @reduce_cls_vec
CUDA_VISIBLE_DEVICES=5 python3 $server/GOmultitask/GCN/encoder/do_model.py --lr 0.001 --main_dir $work_dir --qnli_dir $data_dir --batch_size_aa_go 24 --result_folder $result_folder --epoch 100 --use_cuda --metric_option $metric_option --nonlinear_gcnn $nonlinear_gcnn --def_emb_dim $def_emb_dim --gcnn_dim $gcnn_dim > $result_folder/train3.log



### use DEEPGO data
## use cosine similarity as objective function 
def_emb_dim='768'
gcnn_dim='768'
metric_option='cosine'
nonlinear_gcnn='relu'

data_where_train_is='goAndGeneAnnotationMar2017' ## where the train.csv is... notice we can use the same input, we will just remove GO that we don't see in the go.obo in 2016
which_data='deepgo/data' ## where the go.obo is 
server='/local/datdb'
work_dir=$server/$which_data
data_dir=$server/$data_where_train_is/'entailment_data/AicScore/go_bert_cls'
mkdir $work_dir/'GCN'

result_folder=$work_dir/'GCN/'$metric_option'.'$def_emb_dim #$def_emb_dim.'clsVec'
mkdir $result_folder

conda activate tensorflow_gpuenv
cd $server/GOmultitask

## notice we do not add the reduce-flag to call the linear layer in cosine-module. because this would not be the original implementation of GCN
## NOT USE @reduce_cls_vec
CUDA_VISIBLE_DEVICES=5 python3 $server/GOmultitask/GCN/encoder/do_model.py --lr 0.001 --main_dir $work_dir --qnli_dir $data_dir --batch_size_aa_go 24 --result_folder $result_folder --epoch 100 --use_cuda --metric_option $metric_option --nonlinear_gcnn $nonlinear_gcnn --def_emb_dim $def_emb_dim --gcnn_dim $gcnn_dim > $result_folder/train3.log


