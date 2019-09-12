

## run on deepgo sequence only + flat (no maxpool over child nodes), this requires 3 runs, bp, mf, cc
## run fast with if we don't update go definitions embedding 

conda activate tensorflow_gpuenv

def_emb_dim='300'
metric_option='cosine' ## LOAD BACK COSINE
nonlinear_gcnn='relu'
prot_interact_vec_dim='0'

server='/local/datdb'
# work_dir=$server/'goAndGeneAnnotationDec2018'
work_dir=$server/'deepgo/data'

go_enc_model_load=$work_dir/'GCN/'$metric_option'.300'/'best_state_dict.pytorch'

for fold in 1 2 3 4 5 ; do 

for ontology in mf cc bp ; do

label_subset_file=$server/'deepgo/data/train/deepgo.'$ontology'.csv'

deepgo_dir=$server/'deepgo/data/train/fold_'$fold
# ontology='mf'

data_dir=$deepgo_dir

model_choice='DeepGOFlatSeqOnly'
result_folder=$data_dir/'FlatSeqOnly.'$ontology
mkdir $result_folder

cd $server/GOmultitask/

CUDA_VISIBLE_DEVICES=5 python3 $server/GOmultitask/ProtSeq2GO/do_model.py --model_choice $model_choice --prot_interact_vec_dim $prot_interact_vec_dim --ontology $ontology --do_kmer --go_vec_dim 0 --prot_vec_dim 832 --optim_choice RMSprop --lr 0.0001 --main_dir $work_dir --data_dir $data_dir --batch_size_label 24 --result_folder $result_folder --epoch 200 --use_cuda --metric_option $metric_option --nonlinear_gcnn $nonlinear_gcnn --go_enc_model_load $go_enc_model_load --label_subset_file $label_subset_file > $result_folder/train.log

done

done 


## run on deepgo, ADD IN PROT-PROT EMB + sequence + flat (no maxpool over child nodes), this requires 3 runs, bp, mf, cc 
## run fast with if we don't update go definitions embedding 

conda activate tensorflow_gpuenv

def_emb_dim='300'
metric_option='cosine' ## LOAD BACK COSINE
nonlinear_gcnn='relu'
prot_interact_vec_dim='256'

server='/local/datdb'
# work_dir=$server/'goAndGeneAnnotationDec2018'
work_dir=$server/'deepgo/data'

go_enc_model_load=$work_dir/'GCN/'$metric_option'.300'/'best_state_dict.pytorch'

for fold in 1 2 3 4 5 ; do 

for ontology in mf cc bp ; do

label_subset_file=$server/'deepgo/data/train/deepgo.'$ontology'.csv'

deepgo_dir=$server/'deepgo/data/train/fold_'$fold
# ontology='mf'

data_dir=$deepgo_dir

model_choice='DeepGOFlatSeqProt'
result_folder=$data_dir/'FlatSeqPpi.'$ontology
mkdir $result_folder

cd $server/GOmultitask/

CUDA_VISIBLE_DEVICES=6 python3 $server/GOmultitask/ProtSeq2GO/do_model.py --model_choice $model_choice --has_ppi_emb --prot_interact_vec_dim $prot_interact_vec_dim --ontology $ontology --do_kmer --go_vec_dim 0 --prot_vec_dim 832 --optim_choice RMSprop --lr 0.0001 --main_dir $work_dir --data_dir $data_dir --batch_size_label 24 --result_folder $result_folder --epoch 100 --use_cuda --metric_option $metric_option --nonlinear_gcnn $nonlinear_gcnn --go_enc_model_load $go_enc_model_load --label_subset_file $label_subset_file > $result_folder/train.log

done

done 



## run on deepgo, ADD our GO encoder, ADD IN PROT-PROT EMB + sequence + flat (no maxpool over child nodes), this requires 3 runs, bp, mf, cc 
## run fast with if we don't update go definitions embedding 

conda activate tensorflow_gpuenv

def_emb_dim='300'
metric_option='cosine' ## LOAD BACK COSINE
nonlinear_gcnn='relu'
prot_interact_vec_dim='256'

server='/local/datdb'
# work_dir=$server/'goAndGeneAnnotationDec2018'
work_dir=$server/'deepgo/data'

go_enc_model_load=$work_dir/'GCN/'$metric_option'.300'/'best_state_dict.pytorch'

for fold in 1 2 3 4 5 ; do 

for ontology in mf cc bp ; do

label_subset_file=$server/'deepgo/data/train/deepgo.'$ontology'.csv'

deepgo_dir=$server/'deepgo/data/train/fold_'$fold
# ontology='mf'

data_dir=$deepgo_dir

model_choice='DeepGOFlatSeqProtHwayGo'
result_folder=$data_dir/'FlatSeqProtHwayGo.'$ontology
mkdir $result_folder

cd $server/GOmultitask/

CUDA_VISIBLE_DEVICES=7 python3 $server/GOmultitask/ProtSeq2GO/do_model.py --model_choice $model_choice --has_ppi_emb --prot_interact_vec_dim $prot_interact_vec_dim --ontology $ontology --do_kmer --go_vec_dim 300 --prot_vec_dim 832 --optim_choice RMSprop --lr 0.0001 --main_dir $work_dir --data_dir $data_dir --batch_size_label 24 --result_folder $result_folder --epoch 100 --use_cuda --metric_option $metric_option --nonlinear_gcnn $nonlinear_gcnn --go_enc_model_load $go_enc_model_load --label_subset_file $label_subset_file --fix_go_emb > $result_folder/train.log

done

done 


### **** run seq + prot interact net + tree

conda activate tensorflow_gpuenv

def_emb_dim='300'
metric_option='cosine' ## LOAD BACK COSINE
nonlinear_gcnn='relu'
prot_interact_vec_dim='256'

server='/local/datdb'
# work_dir=$server/'goAndGeneAnnotationDec2018'
work_dir=$server/'deepgo/data'

go_enc_model_load=$work_dir/'GCN/'$metric_option'.300'/'best_state_dict.pytorch'


for fold in 1 2 3 4 5 ; do 

for ontology in mf cc bp ; do

adjacency_matrix=$work_dir/'adjacency_matrix_'$ontology
label_in_ontology=$work_dir/'terms_in_'$ontology'.csv'

label_subset_file=$server/'deepgo/data/train/deepgo.'$ontology'.csv'

deepgo_dir=$server/'deepgo/data/train/fold_'$fold
# ontology='mf'

data_dir=$deepgo_dir

model_choice='DeepGOTreeSeqProt'
result_folder=$data_dir/'TreeSeqProt.'$ontology
mkdir $result_folder

cd $server/GOmultitask/

CUDA_VISIBLE_DEVICES=5 python3 $server/GOmultitask/ProtSeq2GO/do_model.py --tree --model_choice $model_choice --has_ppi_emb --prot_interact_vec_dim $prot_interact_vec_dim --ontology $ontology --do_kmer --go_vec_dim 0 --prot_vec_dim 832 --optim_choice RMSprop --lr 0.0001 --main_dir $work_dir --data_dir $data_dir --batch_size_label 24 --result_folder $result_folder --epoch 200 --use_cuda --metric_option $metric_option --nonlinear_gcnn $nonlinear_gcnn --go_enc_model_load $go_enc_model_load --label_subset_file $label_subset_file --fix_go_emb --adjacency_matrix $adjacency_matrix --label_in_ontology $label_in_ontology > $result_folder/trainSGD.log

done

done 



### **** run go encoder + seq + prot interact net + tree

conda activate tensorflow_gpuenv

def_emb_dim='300'
metric_option='cosine' ## LOAD BACK COSINE
nonlinear_gcnn='relu'
prot_interact_vec_dim='256'

server='/local/datdb'
# work_dir=$server/'goAndGeneAnnotationDec2018'
work_dir=$server/'deepgo/data'

go_enc_model_load=$work_dir/'GCN/'$metric_option'.300'/'best_state_dict.pytorch'


for fold in 1 2 3 4 5 ; do 

for ontology in mf cc bp ; do

adjacency_matrix=$work_dir/'adjacency_matrix_'$ontology
label_in_ontology=$work_dir/'terms_in_'$ontology'.csv'

label_subset_file=$server/'deepgo/data/train/deepgo.'$ontology'.csv'

deepgo_dir=$server/'deepgo/data/train/fold_'$fold
# ontology='mf'

data_dir=$deepgo_dir

model_choice='DeepGOTreeSeqProtHwayGo'
result_folder=$data_dir/'TreeSeqProtHwayGcnGo.'$ontology
mkdir $result_folder

cd $server/GOmultitask/

CUDA_VISIBLE_DEVICES=4 python3 $server/GOmultitask/ProtSeq2GO/do_model.py --tree --model_choice $model_choice --has_ppi_emb --prot_interact_vec_dim $prot_interact_vec_dim --ontology $ontology --do_kmer --go_vec_dim 300 --prot_vec_dim 832 --optim_choice RMSprop --lr 0.0001 --main_dir $work_dir --data_dir $data_dir --batch_size_label 24 --result_folder $result_folder --epoch 200 --use_cuda --metric_option $metric_option --nonlinear_gcnn $nonlinear_gcnn --go_enc_model_load $go_enc_model_load --label_subset_file $label_subset_file --fix_go_emb --adjacency_matrix $adjacency_matrix --label_in_ontology $label_in_ontology > $result_folder/train.log

done

done 


### run seq only + tree

def_emb_dim='300'
metric_option='cosine' ## LOAD BACK COSINE
nonlinear_gcnn='relu'

server='/local/datdb'

work_dir=$server/'deepgo/data'

deepgo_dir=$server/'deepgo/data/train'
# ontology='mf'

for ontology in cc bp ; do

label_subset_file=$deepgo_dir/'deepgo.'$ontology'.csv'
data_dir=$deepgo_dir

prot_interact_vec_dim='0'
model_choice='DeepGOTreeSeqOnly'
result_folder=$data_dir/'TreeSeqOnly.'$ontology
mkdir $result_folder

adjacency_matrix=$work_dir/'adjacency_matrix_'$ontology
label_in_ontology=$work_dir/'terms_in_'$ontology'.csv'

# prot2seq_model_load=$result_folder/'best_state_dict.pytorch'

go_enc_model_load=$work_dir/'GCN/'$metric_option'.300'/'best_state_dict.pytorch'

conda activate tensorflow_gpuenv
cd $server/GOmultitask/

# args.fix_go_emb

CUDA_VISIBLE_DEVICES=6 python3 $server/GOmultitask/ProtSeq2GO/do_model.py --tree --model_choice $model_choice --prot_interact_vec_dim $prot_interact_vec_dim --ontology $ontology --do_kmer --go_vec_dim 0 --prot_vec_dim 832 --lr 0.001 --main_dir $work_dir --data_dir $data_dir --batch_size_label 16 --result_folder $result_folder --epoch 200 --use_cuda --metric_option $metric_option --nonlinear_gcnn $nonlinear_gcnn --go_enc_model_load $go_enc_model_load --label_subset_file $label_subset_file --fix_go_emb --adjacency_matrix $adjacency_matrix --label_in_ontology $label_in_ontology --prot2seq_model_load $prot2seq_model_load > $result_folder/train.log

done

