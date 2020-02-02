
conda activate tensorflow_gpuenv

def_emb_dim=768 ####'0' ## doesn't matter, just dummy, baseline doesn't use go vector
metric_option='cosine' ## LOAD BACK COSINE
prot_interact_vec_dim='256'

server='/local/datdb'

w2v_emb=$server/'deepgo/data/word_pubmed_intersect_GOdb_w2v_pretrained.pickle'
vocab_list=$server/'deepgo/data/word_pubmed_intersect_GOdb.txt'

work_dir=$server/'deepgo/data' ## should probably not need this if we don't hardcode extra needed stuffs.
batch_size_aa_go=32 #24 # 32
# sgd_lr=0.05
sgd_lr=0.0001
optim_choice='RMSprop'

for lr in 0.0005 ; do
  for fold in 1 ; do
    for ontology in mf cc bp ; do # cc bp bp mf cc

      label_subset_file=$server/'deepgo/data/train/deepgo.'$ontology'.csv'
      label_counter_dict=$server/'deepgo/data/train/fold_1/CountGoInTrain-'$ontology'.pickle'
      deepgo_dir=$server/'deepgo/data/train/fold_'$fold
      data_dir=$deepgo_dir
      # model_choice='DeepGOFlatSeqProtHwayNotUseGo' 
      model_choice='DeepGOFlatSeqProtHwayNotUseGo' ###  ## DeepGOFlatSeqProt

      if [[ $model_choice == 'DeepGOFlatSeqProt' ]]
      then
      def_emb_dim=0 ##!! so we don't load protein vec
      fi

      # result_folder='/local/datdb/deepgo/data/train/fold_1/FlatSeqProtBaseline'
      result_folder=$data_dir/$model_choice'Base'
      mkdir $result_folder
      result_folder=$data_dir/$model_choice'Base/Interact'
      mkdir $result_folder

      result_folder=$result_folder/$ontology'b'$batch_size_aa_go'lr'$lr$optim_choice # 'SwSgd' ## sub folder
      # result_folder=$result_folder/$ontology'b32lr0.001' # 'SwSgd' ## sub folder
      # prot2seq_model_load=$result_folder/'best_state_dict.pytorch'
      # result_folder=$data_dir/'Baseline2NoPPI.'$def_emb_dim ##!! re-define
      # result_folder=$result_folder/$ontology'b'$batch_size_aa_go'lr'$lr'continue'

      ## COMMENT train
      cd $server/GOmultitask/
      mkdir $result_folder
      ## !! calling @do_model_bilstm doesn't matter, because it will just do whatever the model @model_choice does
      CUDA_VISIBLE_DEVICES=4 python3 $server/GOmultitask/ProtSeq2GO/do_model_bilstm.py $extra_linear_layer --sgd_lr $sgd_lr --bilstm_dim 1024 --vocab_list $vocab_list --w2v_emb $w2v_emb --model_choice $model_choice --has_ppi_emb --prot_interact_vec_dim $prot_interact_vec_dim --ontology $ontology --do_kmer --go_vec_dim $def_emb_dim --prot_vec_dim 832 --optim_choice $optim_choice --lr $lr --main_dir $work_dir --data_dir $data_dir --batch_size_aa_go $batch_size_aa_go --result_folder $result_folder --epoch 50 --use_cuda --metric_option $metric_option --label_subset_file $label_subset_file --fix_go_emb --def_emb_dim $def_emb_dim --reduce_cls_vec --switch_sgd > $result_folder/train.log ## --prot2seq_model_load $prot2seq_model_load --go_enc_model_load $go_enc_model_load --label_in_ontology $label_in_ontology

      ## COMMENT test
      prot2seq_model_load=$result_folder/'best_state_dict.pytorch'
      CUDA_VISIBLE_DEVICES=4 python3 $server/GOmultitask/ProtSeq2GO/do_model_bilstm.py $extra_linear_layer --bilstm_dim 1024 --vocab_list $vocab_list --w2v_emb $w2v_emb --model_choice $model_choice --has_ppi_emb --prot_interact_vec_dim $prot_interact_vec_dim --ontology $ontology --do_kmer --go_vec_dim $def_emb_dim --prot_vec_dim 832 --main_dir $work_dir --data_dir $data_dir --batch_size_aa_go 64 --result_folder $result_folder --use_cuda --metric_option $metric_option --label_subset_file $label_subset_file --fix_go_emb --def_emb_dim $def_emb_dim --reduce_cls_vec --prot2seq_model_load $prot2seq_model_load --not_train > $result_folder/test_frequency.log

    done
  done
done
