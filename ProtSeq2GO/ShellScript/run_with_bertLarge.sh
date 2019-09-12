

##### !!!!!! 


## run on deepgo, ADD our GO encoder, ADD IN PROT-PROT EMB + sequence + flat (no maxpool over child nodes), this requires 3 runs, bp, mf, cc 
## run fast with if we don't update go definitions embedding 

conda activate tensorflow_gpuenv

def_emb_dim='768'
metric_option='cosine' ## LOAD BACK COSINE

prot_interact_vec_dim='256'
server='/local/datdb'
work_dir=$server/'deepgo/data'
bert_model=$server/'goAndGeneAnnotationMar2017/BERT_base_cased_tune_go_branch/fine_tune_lm_bioBERT' # use the full mask + nextSentence to innit
## this is bert train with cosine distance, it can be saved at different folder than @bert_model
bert_already_train=$work_dir/'cosine.Cls768.Linear768'
go_enc_model_load=$bert_already_train/'best_state_dict.pytorch'

## REDEFINE
work_dir=$server/'deepgo/dataExpandGoSet' ## should probably not need this if we don't hardcode extra needed stuffs.
for fold in 1 ; do 
  for ontology in mf cc bp ; do

  # label_in_ontology=$work_dir/'terms_in_'$ontology'.csv'
  label_subset_file=$server/'deepgo/dataExpandGoSet/train/deepgo.'$ontology'.csv'
  label_counter_dict=$server/'deepgo/dataExpandGoSet/train/CountGoInTrain-'$ontology'.pickle'
  deepgo_dir=$server/'deepgo/dataExpandGoSet/train/fold_'$fold
  data_dir=$deepgo_dir

  model_choice='DeepGOFlatSeqProtHwayGo'
  result_folder=$data_dir/'FlatSeqProtHwayBert768Lr0.001b64.'$ontology
  # prot2seq_model_load=$result_folder/'best_state_dict.pytorch'

  # ## continue training save to a new place
  # result_folder=$data_dir/'FlatSeqProtHwayBert768R2.'$ontology
  mkdir $result_folder

  cd $server/GOmultitask/

  ## rmsprop 0.001 batch 32 --> sgd
  ## for bp, rmsprop 0.001 batch 32 --> sgd doesn't help, it seem step size is too large ?? 

  CUDA_VISIBLE_DEVICES=5 python3 $server/GOmultitask/ProtSeq2GO/do_model_bert.py --model_choice $model_choice --has_ppi_emb --prot_interact_vec_dim $prot_interact_vec_dim --ontology $ontology --do_kmer --go_vec_dim $def_emb_dim --prot_vec_dim 832 --optim_choice RMSprop --lr 0.001 --main_dir $work_dir --data_dir $data_dir --batch_size_label 64 --result_folder $result_folder --epoch 40 --use_cuda --metric_option $metric_option --go_enc_model_load $go_enc_model_load --label_subset_file $label_subset_file --fix_go_emb --bert_model $bert_model --def_emb_dim $def_emb_dim --reduce_cls_vec --label_counter_dict $label_counter_dict > $result_folder/train.log 

  # --prot2seq_model_load $prot2seq_model_load --label_in_ontology $label_in_ontology

  prot2seq_model_load=$result_folder/'best_state_dict.pytorch'

  CUDA_VISIBLE_DEVICES=5 python3 $server/GOmultitask/ProtSeq2GO/do_model_bert.py --model_choice $model_choice --has_ppi_emb --prot_interact_vec_dim $prot_interact_vec_dim --ontology $ontology --do_kmer --go_vec_dim $def_emb_dim --prot_vec_dim 832 --main_dir $work_dir --data_dir $data_dir --batch_size_label 64 --result_folder $result_folder --epoch 0 --use_cuda --metric_option $metric_option --go_enc_model_load $go_enc_model_load --label_subset_file $label_subset_file --fix_go_emb --bert_model $bert_model --def_emb_dim $def_emb_dim --reduce_cls_vec --prot2seq_model_load $prot2seq_model_load --not_train --label_counter_dict $label_counter_dict > $result_folder/test_frequency.log

  # --label_in_ontology $label_in_ontology
  done
done

