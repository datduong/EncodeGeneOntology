
## run on deepgo, ADD our GO encoder, ADD IN PROT-PROT EMB + sequence + flat (no maxpool over child nodes), this requires 3 runs, bp, mf, cc 
## run fast with if we don't update go definitions embedding 

conda activate tensorflow_gpuenv

def_emb_dim='300'
metric_option='cosine' ## LOAD BACK COSINE

prot_interact_vec_dim='256'

server='/local/datdb'


work_dir=$server/'deepgo/data'

bert_model=$server/'goAndGeneAnnotationDec2018/BERT_base_cased_tune_go_branch/fine_tune_lm_bioBERT' # use the full mask + nextSentence to innit

## this is bert train with cosine distance, it can be saved at different folder than @bert_model
bert_already_train=$work_dir/'cosine.768.reduce300ClsVec'
go_enc_model_load=$bert_already_train/'best_state_dict.pytorch'

for fold in 1 ; do 

for ontology in mf cc bp ; do

label_in_ontology=$work_dir/'terms_in_'$ontology'.csv'

label_subset_file=$server/'deepgo/data/train/deepgo.'$ontology'.csv'

deepgo_dir=$server/'deepgo/data/train/fold_'$fold
# ontology='mf'

data_dir=$deepgo_dir

model_choice='DeepGOFlatSeqProtHwayGo'
result_folder=$data_dir/'FlatSeqProtHwayBertGo.'$ontology
mkdir $result_folder

cd $server/GOmultitask/

CUDA_VISIBLE_DEVICES=7 python3 $server/GOmultitask/ProtSeq2GO/do_model_bert.py --model_choice $model_choice --has_ppi_emb --prot_interact_vec_dim $prot_interact_vec_dim --ontology $ontology --do_kmer --go_vec_dim 300 --prot_vec_dim 832 --optim_choice RMSprop --lr 0.0001 --main_dir $work_dir --data_dir $data_dir --batch_size_label 24 --result_folder $result_folder --epoch 100 --use_cuda --metric_option $metric_option --go_enc_model_load $go_enc_model_load --label_subset_file $label_subset_file --label_in_ontology $label_in_ontology --fix_go_emb --bert_model $bert_model --def_emb_dim $def_emb_dim --reduce_cls_vec > $result_folder/train.log

done

done 

