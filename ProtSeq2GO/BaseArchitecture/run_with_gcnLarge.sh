

conda activate tensorflow_gpuenv

def_emb_dim='768'
gcnn_dim=$def_emb_dim
metric_option='cosine' ## LOAD BACK COSINE
nonlinear_gcnn='relu'
prot_interact_vec_dim='256'

server='/local/datdb'
work_dir=$server/'deepgo/data'
go_enc_model_load=$work_dir/'GCN/'$metric_option'.768'/'best_state_dict.pytorch'

#### do not have vectors written out for GCN

work_dir=$server/'deepgo/dataExpandGoSet' #### should probably not need this if we don't hardcode extra needed stuffs.

batch_size_aa_go=32

for model_choice in 'DeepGOFlatSeqProtHwayGo' 'DeepGOFlatSeqProtHwayGoNotUsePPI'; do 

  prot_interact_vec_dim=256
  if [[ $model_choice == 'DeepGOFlatSeqProtHwayGoNotUsePPI' ]]
  then
    prot_interact_vec_dim=0 ##!! so we don't load protein vec
  fi

  for lr in 0.001 ; do
    for fold in 1 ; do
      for ontology in mf cc bp ; do # bp mf cc ;

        label_subset_file=$server/'deepgo/dataExpandGoSet/train/deepgo.'$ontology'.csv'
        label_counter_dict=$server/'deepgo/dataExpandGoSet/train/CountGoInTrain-'$ontology'.pickle'
        deepgo_dir=$server/'deepgo/dataExpandGoSet/train/fold_'$fold
        data_dir=$deepgo_dir

        # # model_choice='DeepGOFlatSeqProtHwayGo' ##!!
        # model_choice='DeepGOFlatSeqProtHwayGoNotUsePPI'  

        result_folder=$data_dir/$model_choice'GCN'
        mkdir $result_folder

        result_folder=$result_folder/$ontology'b'$batch_size_aa_go'lr'$lr # 'SwSgd' ## sub folder
        # result_folder=$result_folder/$ontology'b32lr0.001' # 'SwSgd' ## sub folder
        # prot2seq_model_load=$result_folder/'best_state_dict.pytorch'
        # result_folder=$data_dir/'GCN.'$def_emb_dim ## re-define
        # result_folder=$result_folder/$ontology'b'$batch_size_aa_go'lr'$lr'continue'

        cd $server/GOmultitask/
        mkdir $result_folder
        CUDA_VISIBLE_DEVICES=0 python3 $server/GOmultitask/ProtSeq2GO/do_model_gcn.py --add_name same-origin --model_choice $model_choice --has_ppi_emb --prot_interact_vec_dim $prot_interact_vec_dim --ontology $ontology --do_kmer --gcnn_dim $gcnn_dim --def_emb_dim $def_emb_dim --go_vec_dim $def_emb_dim --prot_vec_dim 832 --optim_choice RMSprop --lr $lr --main_dir $work_dir --data_dir $data_dir --batch_size_aa_go $batch_size_aa_go --result_folder $result_folder --epoch 100 --use_cuda --metric_option $metric_option --nonlinear_gcnn $nonlinear_gcnn --go_enc_model_load $go_enc_model_load --label_subset_file $label_subset_file --fix_go_emb --label_counter_dict $label_counter_dict --switch_sgd > $result_folder/train.log

        # --prot2seq_model_load $prot2seq_model_load --switch_sgd 

        ##!! test
        prot2seq_model_load=$result_folder/'best_state_dict.pytorch'
        CUDA_VISIBLE_DEVICES=0 python3 $server/GOmultitask/ProtSeq2GO/do_model_gcn.py --add_name same-origin --model_choice $model_choice --has_ppi_emb --prot_interact_vec_dim $prot_interact_vec_dim --ontology $ontology --do_kmer --gcnn_dim $gcnn_dim --def_emb_dim $def_emb_dim --go_vec_dim $def_emb_dim --prot_vec_dim 832 --optim_choice RMSprop --main_dir $work_dir --data_dir $data_dir --batch_size_aa_go 64 --result_folder $result_folder --epoch 0 --use_cuda --metric_option $metric_option --nonlinear_gcnn $nonlinear_gcnn --go_enc_model_load $go_enc_model_load --label_subset_file $label_subset_file --fix_go_emb --prot2seq_model_load $prot2seq_model_load --not_train --label_counter_dict $label_counter_dict > $result_folder/test_frequency.log

      done
    done
  done

done
