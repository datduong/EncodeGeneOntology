

conda activate tensorflow_gpuenv

def_emb_dim='768'
metric_option='cosine'
prot_interact_vec_dim='256'
server='/local/datdb'
work_dir=$server/'deepgo/data'
bert_model=$server/'goAndGeneAnnotationMar2017/BERT_base_cased_tune_go_branch/fine_tune_lm_bioBERT' # use the full mask + nextSentence to innit
## this is bert train with cosine distance, it can be saved at different folder than @bert_model
bert_already_train=$work_dir/'cosine.Cls768.Linear768' ####
go_enc_model_load=$bert_already_train/'best_state_dict.pytorch'

batch_size_aa_go=32 #24 # 32
weight_decay=0
sgd_lr=0.05
OptimChoice='Adam'

for model_choice in 'DeepGOFlatSeqConcatGo'; do # 'DeepGOFlatSeqProtHwayGo'

  prot_interact_vec_dim=256
  if [[ $model_choice == 'DeepGOFlatSeqProtHwayGoNotUsePPI' ]]
  then
    prot_interact_vec_dim=0 ##!! so we don't load protein vec
  fi

  if [[ $model_choice == 'DeepGOFlatSeqConcatGo' ]]
  then
    prot_interact_vec_dim=0 ##!! so we don't load protein vec
    sgd_lr=0.0001
  fi

  for lr in 0.0005 ; do
    for fold in 1 ; do
      for ontology in mf cc bp ; do # cc bp bp mf cc

      label_in_ontology=$work_dir/'terms_in_'$ontology'.csv'
      label_subset_file=$server/'deepgo/data/train/deepgo.'$ontology'.csv'
      deepgo_dir=$server/'deepgo/data/train/fold_'$fold
      data_dir=$deepgo_dir
      label_counter_dict=$deepgo_dir/'CountGoInTrain-'$ontology'.pickle'

      result_folder=$data_dir/$model_choice'BertCLS12'
      mkdir $result_folder

      result_folder=$result_folder/$ontology'b'$batch_size_aa_go'lr'$lr # 'SwSgd' ## sub folder
      # result_folder=$result_folder/$ontology'b32lr0.001' # 'SwSgd' ## sub folder
      # prot2seq_model_load=$result_folder/'best_state_dict.pytorch'
      # result_folder=$data_dir/'GCN.'$def_emb_dim ## re-define
      # result_folder=$result_folder/$ontology'b'$batch_size_aa_go'lr'$lr'continue'

      ## COMMENT testing
      cd $server/GOmultitask/
      mkdir $result_folder
      CUDA_VISIBLE_DEVICES=2 python3 $server/GOmultitask/ProtSeq2GO/do_model_bert.py --model_choice $model_choice --has_ppi_emb --prot_interact_vec_dim $prot_interact_vec_dim --ontology $ontology --do_kmer --go_vec_dim $def_emb_dim --prot_vec_dim 832 --optim_choice RMSprop --lr $lr --main_dir $work_dir --data_dir $data_dir --batch_size_aa_go $batch_size_aa_go --result_folder $result_folder --epoch 100 --use_cuda --metric_option $metric_option --go_enc_model_load $go_enc_model_load --label_subset_file $label_subset_file --label_in_ontology $label_in_ontology --fix_go_emb --bert_model $bert_model --def_emb_dim $def_emb_dim --reduce_cls_vec --label_counter_dict $label_counter_dict --switch_sgd > $result_folder/train.log

      ## COMMENT testing
      prot2seq_model_load=$result_folder/'best_state_dict.pytorch'
      CUDA_VISIBLE_DEVICES=2 python3 $server/GOmultitask/ProtSeq2GO/do_model_bert.py --model_choice $model_choice --has_ppi_emb --prot_interact_vec_dim $prot_interact_vec_dim --ontology $ontology --do_kmer --go_vec_dim $def_emb_dim --prot_vec_dim 832 --optim_choice RMSprop --main_dir $work_dir --data_dir $data_dir --batch_size_aa_go 64 --result_folder $result_folder --epoch 0 --use_cuda --metric_option $metric_option --go_enc_model_load $go_enc_model_load --label_subset_file $label_subset_file --label_in_ontology $label_in_ontology --fix_go_emb --bert_model $bert_model --def_emb_dim $def_emb_dim --reduce_cls_vec --prot2seq_model_load $prot2seq_model_load --not_train --label_counter_dict $label_counter_dict > $result_folder/test_frequency.log

       #### test zeroshot
      prot2seq_model_load=$result_folder/'best_state_dict.pytorch' ##!! define again
      label_subset_file=$server/'deepgo/dataExpandGoSet/train/deepgo.'$ontology'.csv'
      label_counter_dict=$server/'deepgo/dataExpandGoSet/train/fold_1/CountGoInTrain-'$ontology'.pickle'
      test_data_name=$server/'deepgo/dataExpandGoSet/train/fold_'$fold/'test-'$ontology'-same-origin.tsv'
      test_output_name=$result_folder/'test-'$ontology'-same-origin'
      CUDA_VISIBLE_DEVICES=2 python3 $server/GOmultitask/ProtSeq2GO/do_model_bert.py --test_data_name $test_data_name --test_output_name $test_output_name --model_choice $model_choice --has_ppi_emb --prot_interact_vec_dim $prot_interact_vec_dim --ontology $ontology --do_kmer --go_vec_dim $def_emb_dim --prot_vec_dim 832 --optim_choice RMSprop --main_dir $work_dir --data_dir $data_dir --batch_size_aa_go 48 --result_folder $result_folder --epoch 0 --use_cuda --metric_option $metric_option --go_enc_model_load $go_enc_model_load --label_subset_file $label_subset_file --label_in_ontology $label_in_ontology --fix_go_emb --bert_model $bert_model --def_emb_dim $def_emb_dim --reduce_cls_vec --prot2seq_model_load $prot2seq_model_load --not_train --label_counter_dict $label_counter_dict > $result_folder/own_test_frequency.log

      done
    done
  done
done

