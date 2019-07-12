

## use cosine similarity as objective function 
def_emb_dim='300'
metric_option='cosine'
nonlinear_gcnn='relu'

server='/local/datdb'
work_dir=$server/'goAndGeneAnnotationDec2018'

data_dir=$work_dir/'ecoli_yeast_human/full_is_a_fold/fold_1'
result_folder=$data_dir/'GCN'$metric_option'.300.FixProt'
mkdir $result_folder

model_load=$work_dir/'GCN/'$metric_option'.300'/'best_state_dict.pytorch'

conda activate tensorflow_gpuenv
cd $server/GOmultitask/

CUDA_VISIBLE_DEVICES=1 python3 $server/GOmultitask/ProtSeq2GO/do_model.py --lr 0.001 --main_dir $work_dir --data_dir $data_dir --batch_size_label 12 --result_folder $result_folder --epoch 100 --use_cuda --metric_option $metric_option --nonlinear_gcnn $nonlinear_gcnn --def_emb_dim $def_emb_dim --model_load $model_load --fix_prot_emb > $result_folder/train.log



## use cosine similarity as objective function
def_emb_dim='300'
metric_option='cosine'
nonlinear_gcnn='relu'

server='/local/datdb'
work_dir=$server/'goAndGeneAnnotationDec2018'

data_dir=$work_dir/'ecoli_yeast_human/full_is_a_fold/fold_1'
result_folder=$data_dir/'GCN'$metric_option'.300.FixProtFixGo'
mkdir $result_folder

model_load=$work_dir/'GCN/'$metric_option'.300'/'best_state_dict.pytorch'

conda activate tensorflow_gpuenv
cd $server/GOmultitask/

CUDA_VISIBLE_DEVICES=3 python3 $server/GOmultitask/ProtSeq2GO/do_model.py --lr 0.001 --main_dir $work_dir --data_dir $data_dir --batch_size_label 12 --result_folder $result_folder --epoch 100 --use_cuda --metric_option $metric_option --nonlinear_gcnn $nonlinear_gcnn --def_emb_dim $def_emb_dim --model_load $model_load --fix_prot_emb --fix_go_emb > $result_folder/train.log


## run on deepgo sequence only + flat (no maxpool over child nodes), this requires 3 runs, bp, mf, cc

def_emb_dim='300'
metric_option='cosine' ## LOAD BACK COSINE
nonlinear_gcnn='relu'

server='/local/datdb'
work_dir=$server/'goAndGeneAnnotationDec2018'

deepgo_dir=$server/'deepgo/data/train'
ontology='mf'
label_subset_file=$deepgo_dir/'deepgo.'$ontology'.csv'
data_dir=$deepgo_dir

result_folder=$data_dir/'GCN'$metric_option'.300.SeqFlat.'$ontology
mkdir $result_folder

model_load=$work_dir/'GCN/'$metric_option'.300'/'best_state_dict.pytorch'

conda activate tensorflow_gpuenv
cd $server/GOmultitask/

CUDA_VISIBLE_DEVICES=1 python3 $server/GOmultitask/ProtSeq2GO/do_model.py --ontology $ontology --do_kmer --go_vec_dim 0 --prot_vec_dim 832 --lr 0.001 --main_dir $work_dir --data_dir $data_dir --batch_size_label 128 --result_folder $result_folder --epoch 100 --use_cuda --metric_option $metric_option --nonlinear_gcnn $nonlinear_gcnn --model_load $model_load --label_subset_file $label_subset_file > $result_folder/train.log


## run on deepgo sequence only + flat (no maxpool over child nodes), this requires 3 runs, bp, mf, cc ADD IN PROT-PROT EMB
## run fast with if we don't update go definitions embedding 

def_emb_dim='300'
metric_option='cosine' ## LOAD BACK COSINE
nonlinear_gcnn='relu'
prot_interact_vec_dim='256'

server='/local/datdb'
# work_dir=$server/'goAndGeneAnnotationDec2018'
work_dir=$server/'deepgo/data'

deepgo_dir=$server/'deepgo/data/train'
# ontology='mf'

model_load=$work_dir/'GCN/'$metric_option'.300'/'best_state_dict.pytorch'

for ontology in mf cc bp ; do

label_subset_file=$deepgo_dir/'deepgo.'$ontology'.csv'
data_dir=$deepgo_dir

result_folder=$data_dir/'SeqPpiFlat.'$ontology
mkdir $result_folder

conda activate tensorflow_gpuenv
cd $server/GOmultitask/

CUDA_VISIBLE_DEVICES=1 python3 $server/GOmultitask/ProtSeq2GO/do_model.py --has_ppi_emb --prot_interact_vec_dim $prot_interact_vec_dim --ontology $ontology --do_kmer --go_vec_dim 0 --prot_vec_dim 832 --lr 0.001 --main_dir $work_dir --data_dir $data_dir --batch_size_label 16 --result_folder $result_folder --epoch 100 --use_cuda --metric_option $metric_option --nonlinear_gcnn $nonlinear_gcnn --model_load $model_load --label_subset_file $label_subset_file > $result_folder/train.log

done


## ADD GO encoder

def_emb_dim='300'
metric_option='cosine' 
nonlinear_gcnn='relu'

server='/local/datdb'

work_dir=$server/'deepgo/data'

deepgo_dir=$server/'deepgo/data/train'
ontology='mf'
label_subset_file=$deepgo_dir/'deepgo.'$ontology'.csv'
data_dir=$deepgo_dir

prot_interact_vec_dim='256'
model_choice='DeepGOFlatSeqProtHwayGo'
result_folder=$data_dir/'FlatSeqProtHwayGcnGo.'$ontology
mkdir $result_folder

# prot2seq_model_load=$result_folder/'best_state_dict.pytorch'

go_enc_model_load=$work_dir/'GCN/'$metric_option'.300'/'best_state_dict.pytorch'

conda activate tensorflow_gpuenv
cd $server/GOmultitask/

# args.fix_go_emb

CUDA_VISIBLE_DEVICES=3 python3 $server/GOmultitask/ProtSeq2GO/do_model.py --model_choice $model_choice --has_ppi_emb --prot_interact_vec_dim $prot_interact_vec_dim --ontology $ontology --do_kmer --go_vec_dim 300 --prot_vec_dim 832 --lr 0.001 --main_dir $work_dir --data_dir $data_dir --batch_size_label 32 --result_folder $result_folder --epoch 200 --use_cuda --metric_option $metric_option --nonlinear_gcnn $nonlinear_gcnn --go_enc_model_load $go_enc_model_load --label_subset_file $label_subset_file --fix_go_emb > $result_folder/train.log



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



### run seq + prot interact net + tree

def_emb_dim='300'
metric_option='cosine' ## LOAD BACK COSINE
nonlinear_gcnn='relu'

server='/local/datdb'

work_dir=$server/'deepgo/data'

deepgo_dir=$server/'deepgo/data/train'
# ontology='mf'

for ontology in mf ; do

label_subset_file=$deepgo_dir/'deepgo.'$ontology'.csv'
data_dir=$deepgo_dir

prot_interact_vec_dim='256'
model_choice='DeepGOTreeSeqProt'
result_folder=$data_dir/'TreeSeqProt.repeat.'$ontology
mkdir $result_folder

adjacency_matrix=$work_dir/'adjacency_matrix_'$ontology
label_in_ontology=$work_dir/'terms_in_'$ontology'.csv'

# prot2seq_model_load=$result_folder/'best_state_dict.pytorch'

go_enc_model_load=$work_dir/'GCN/'$metric_option'.300'/'best_state_dict.pytorch'

conda activate tensorflow_gpuenv
cd $server/GOmultitask/

# args.fix_go_emb 

CUDA_VISIBLE_DEVICES=5 python3 $server/GOmultitask/ProtSeq2GO/do_model.py --tree --model_choice $model_choice --has_ppi_emb --prot_interact_vec_dim $prot_interact_vec_dim --ontology $ontology --do_kmer --go_vec_dim 0 --prot_vec_dim 832 --lr 0.0001 --main_dir $work_dir --data_dir $data_dir --batch_size_label 48 --result_folder $result_folder --epoch 200 --use_cuda --metric_option $metric_option --nonlinear_gcnn $nonlinear_gcnn --go_enc_model_load $go_enc_model_load --label_subset_file $label_subset_file --fix_go_emb --adjacency_matrix $adjacency_matrix --label_in_ontology $label_in_ontology > $result_folder/train.log

done


### run seq + prot interact net + tree + go encoder 

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

prot_interact_vec_dim='256'
model_choice='DeepGOTreeSeqProtHwayGo'
result_folder=$data_dir/'TreeSeqProtHwayGcnGo.'$ontology
mkdir $result_folder

adjacency_matrix=$work_dir/'adjacency_matrix_'$ontology
label_in_ontology=$work_dir/'terms_in_'$ontology'.csv'

# prot2seq_model_load=$result_folder/'best_state_dict.pytorch'

go_enc_model_load=$work_dir/'GCN/'$metric_option'.300'/'best_state_dict.pytorch'

conda activate tensorflow_gpuenv
cd $server/GOmultitask/

# args.fix_go_emb

CUDA_VISIBLE_DEVICES=7 python3 $server/GOmultitask/ProtSeq2GO/do_model.py --tree --model_choice $model_choice --has_ppi_emb --prot_interact_vec_dim $prot_interact_vec_dim --ontology $ontology --do_kmer --go_vec_dim 300 --prot_vec_dim 832 --lr 0.001 --main_dir $work_dir --data_dir $data_dir --batch_size_label 24 --result_folder $result_folder --epoch 200 --use_cuda --metric_option $metric_option --nonlinear_gcnn $nonlinear_gcnn --go_enc_model_load $go_enc_model_load --label_subset_file $label_subset_file --fix_go_emb --adjacency_matrix $adjacency_matrix --label_in_ontology $label_in_ontology > $result_folder/train.log

done
