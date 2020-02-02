
## run on deepgo, ADD our GO encoder, ADD IN PROT-PROT EMB + sequence + flat (no maxpool over child nodes), this requires 3 runs, bp, mf, cc
## run fast with if we don't update go definitions embedding

conda activate tensorflow_gpuenv

def_emb_dim='768'
metric_option='cosine' ## LOAD BACK COSINE
prot_interact_vec_dim='256'

server='/local/datdb'

work_dir=$server/'deepgo/data'
w2v_emb=$server/'deepgo/data/word_pubmed_intersect_GOdb_w2v_pretrained.pickle'
vocab_list=$server/'deepgo/data/word_pubmed_intersect_GOdb.txt'  # all vocab to be used, can be BERT or something, so we don't hardcode
go_enc_model_load=$work_dir/$metric_option'.bilstm768/best_state_dict.pytorch'  ###

## COMMENT redefine @WORK_DIR
work_dir=$server/'deepgo/data' ## should probably not need this if we don't hardcode extra needed stuffs.

add_name='same-origin-bonnie' ##!!##!!##!!##!!##!!##!!
if [[ $add_name == 'same-origin-bonnie' ]]
then
  prot_interact_vec_dim=100
fi


batch_size_aa_go=32
weight_decay=0
optim_choice='RMSprop'
sgd_lr=0.0001

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

        label_subset_file=$server/'deepgo/data/train/deepgo.'$ontology'.csv'
        label_counter_dict=$server/'deepgo/data/train/fold_1/CountGoInTrain-'$ontology'.pickle'
        deepgo_dir=$server/'deepgo/data/train/fold_'$fold
        data_dir=$deepgo_dir

        ##!!##!!##!!##!!
        result_folder=$data_dir/$model_choice'BiLSTM' #$add_name 
        mkdir $result_folder

        result_folder=$result_folder/$ontology'b'$batch_size_aa_go'lr'$lr$optim_choice #'d'$weight_decay # 'SwSgd' ## sub folder
        # result_folder=$result_folder/$ontology'b32lr0.001' # 'SwSgd' ## sub folder
        # prot2seq_model_load=$result_folder/'best_state_dict.pytorch'
        # result_folder=$data_dir/'BertCosLinear768Layer11+12OpW2.'$def_emb_dim ## re-define
        # result_folder=$result_folder/$ontology'b'$batch_size_aa_go'lr'$lr'continue'

        ##!! train
        cd $server/GOmultitask/
        mkdir $result_folder
        CUDA_VISIBLE_DEVICES=2 python3 $server/GOmultitask/ProtSeq2GO/do_model_bilstm.py --bilstm_dim 1024 --vocab_list $vocab_list --w2v_emb $w2v_emb --model_choice $model_choice --has_ppi_emb --prot_interact_vec_dim $prot_interact_vec_dim --ontology $ontology --do_kmer --go_vec_dim $def_emb_dim --prot_vec_dim 832 --optim_choice $optim_choice --lr $lr --main_dir $work_dir --data_dir $data_dir --batch_size_aa_go $batch_size_aa_go --result_folder $result_folder --epoch 100 --use_cuda --metric_option $metric_option --go_enc_model_load $go_enc_model_load --label_subset_file $label_subset_file --fix_go_emb --def_emb_dim $def_emb_dim --reduce_cls_vec --switch_sgd > $result_folder/train.log

        ##!! test
        prot2seq_model_load=$result_folder/'best_state_dict.pytorch'
        CUDA_VISIBLE_DEVICES=2 python3 $server/GOmultitask/ProtSeq2GO/do_model_bilstm.py --bilstm_dim 1024 --vocab_list $vocab_list --w2v_emb $w2v_emb --model_choice $model_choice --has_ppi_emb --prot_interact_vec_dim $prot_interact_vec_dim --ontology $ontology --do_kmer --go_vec_dim $def_emb_dim --prot_vec_dim 832 --main_dir $work_dir --data_dir $data_dir --batch_size_aa_go 48 --result_folder $result_folder --use_cuda --metric_option $metric_option --go_enc_model_load $go_enc_model_load --label_subset_file $label_subset_file --fix_go_emb --def_emb_dim $def_emb_dim --reduce_cls_vec --prot2seq_model_load $prot2seq_model_load --not_train  > $result_folder/test_frequency.log # --prot2seq_model_load $prot2seq_model_load

        #### test zeroshot
        prot2seq_model_load=$result_folder/'best_state_dict.pytorch' ##!! define again
        label_subset_file=$server/'deepgo/dataExpandGoSet/train/deepgo.'$ontology'.csv'
        label_counter_dict=$server/'deepgo/dataExpandGoSet/train/fold_1/CountGoInTrain-'$ontology'.pickle'
        test_data_name=$server/'deepgo/dataExpandGoSet/train/fold_'$fold/'test-'$ontology'-same-origin.tsv'
        test_output_name=$result_folder/'test-'$ontology'-same-origin'
        CUDA_VISIBLE_DEVICES=2 python3 $server/GOmultitask/ProtSeq2GO/do_model_bilstm.py --test_data_name $test_data_name --test_output_name $test_output_name --bilstm_dim 1024 --vocab_list $vocab_list --w2v_emb $w2v_emb --model_choice $model_choice --has_ppi_emb --prot_interact_vec_dim $prot_interact_vec_dim --ontology $ontology --do_kmer --go_vec_dim $def_emb_dim --prot_vec_dim 832 --main_dir $work_dir --data_dir $data_dir --batch_size_aa_go 48 --result_folder $result_folder --use_cuda --metric_option $metric_option --go_enc_model_load $go_enc_model_load --label_subset_file $label_subset_file --fix_go_emb --def_emb_dim $def_emb_dim --reduce_cls_vec --prot2seq_model_load $prot2seq_model_load --not_train  > $result_folder/own_test_frequency.log # 

      done
    done
  done
done


