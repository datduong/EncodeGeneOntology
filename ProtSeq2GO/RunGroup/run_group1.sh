

conda activate tensorflow_gpuenv

#### baseline
def_emb_dim=768 ###'0' ## doesn't matter, just dummy, baseline doesn't use go vector
metric_option='cosine' ## LOAD BACK COSINE
prot_interact_vec_dim='256'
server='/local/datdb'
w2v_emb=$server/'deepgo/data/word_pubmed_intersect_GOdb_w2v_pretrained.pickle'
vocab_list=$server/'deepgo/data/word_pubmed_intersect_GOdb.txt'
##!! ##!! ##!!
work_dir=$server/'deepgo/dataExpandGoSet16Jan2020' ## should probably not need this if we don't hardcode extra needed stuffs.
batch_size_aa_go=32
sgd_lr=0.05
optim_choice='RMSprop'

for model_choice in DeepGOFlatSeqOnly DeepGOFlatSeqProt ; do 

  def_emb_dim=768

  for lr in 0.001 ; do
    for fold in 1 ; do
      for ontology in mf cc bp ; do # cc bp bp mf cc

        label_subset_file=$server/'deepgo/dataExpandGoSet16Jan2020/deepgo.'$ontology'.csv'
        # label_counter_dict=$server/'deepgo/dataExpandGoSet16Jan2020/train/fold_1/CountGoInTrain-'$ontology'.pickle'
        deepgo_dir=$server/'deepgo/dataExpandGoSet16Jan2020/train/fold_'$fold
        data_dir=$deepgo_dir
        # model_choice='DeepGOFlatSeqProtHwayNotUseGo' 
        # model_choice='DeepGOFlatSeqProt' ###  ## DeepGOFlatSeqProt
        # model_choice='DeepGOFlatSeqOnly' ###  ## DeepGOFlatSeqProt

        if [[ $model_choice == 'DeepGOFlatSeqProt' ]]
        then
        def_emb_dim=0 ##!! so we don't load protein vec
        fi

        # result_folder='/local/datdb/deepgo/data/train/fold_1/FlatSeqProtBaseline'
        result_folder=$data_dir/$model_choice'Base'
        mkdir $result_folder
        result_folder=$data_dir/$model_choice'Base/ExactAsIs16Jan20'
        mkdir $result_folder

        result_folder=$result_folder/$ontology'b'$batch_size_aa_go'lr'$lr$optim_choice # 'SwSgd' ## sub folder
        # result_folder=$result_folder/$ontology'b32lr0.001' # 'SwSgd' ## sub folder
        # prot2seq_model_load=$result_folder/'best_state_dict.pytorch'
        # result_folder=$data_dir/'Baseline2NoPPI.'$def_emb_dim ##!! re-define
        # result_folder=$result_folder/$ontology'b'$batch_size_aa_go'lr'$lr'continue'

        ## COMMENT train
        cd $server/GOmultitask/
        mkdir $result_folder

        add_name='same-origin' ####

        #### --extra_linear_layer 

        ## !! calling @do_model_bilstm doesn't matter, because it will just do whatever the model @model_choice does
        CUDA_VISIBLE_DEVICES=2 python3 $server/GOmultitask/ProtSeq2GO/do_model_bilstm.py --add_name same-origin --sgd_lr $sgd_lr --bilstm_dim 1024 --vocab_list $vocab_list --w2v_emb $w2v_emb --model_choice $model_choice --has_ppi_emb --prot_interact_vec_dim $prot_interact_vec_dim --ontology $ontology --do_kmer --go_vec_dim $def_emb_dim --prot_vec_dim 832 --optim_choice $optim_choice --lr $lr --main_dir $work_dir --data_dir $data_dir --batch_size_aa_go $batch_size_aa_go --result_folder $result_folder --epoch 80 --use_cuda --metric_option $metric_option --label_subset_file $label_subset_file --fix_go_emb --def_emb_dim $def_emb_dim --reduce_cls_vec --switch_sgd > $result_folder/train.log ## --prot2seq_model_load $prot2seq_model_load --go_enc_model_load $go_enc_model_load --label_in_ontology $label_in_ontology

        ## COMMENT test

        #### --extra_linear_layer 

        prot2seq_model_load=$result_folder/'best_state_dict.pytorch'
        CUDA_VISIBLE_DEVICES=2 python3 $server/GOmultitask/ProtSeq2GO/do_model_bilstm.py --add_name same-origin --bilstm_dim 1024 --vocab_list $vocab_list --w2v_emb $w2v_emb --model_choice $model_choice --has_ppi_emb --prot_interact_vec_dim $prot_interact_vec_dim --ontology $ontology --do_kmer --go_vec_dim $def_emb_dim --prot_vec_dim 832 --main_dir $work_dir --data_dir $data_dir --batch_size_aa_go 64 --result_folder $result_folder --use_cuda --metric_option $metric_option --label_subset_file $label_subset_file --fix_go_emb --def_emb_dim $def_emb_dim --reduce_cls_vec --prot2seq_model_load $prot2seq_model_load --not_train > $result_folder/test_frequency.log

      done
    done
  done
done 

#### GCN


conda activate tensorflow_gpuenv
def_emb_dim='768'
gcnn_dim=$def_emb_dim
metric_option='cosine' ## LOAD BACK COSINE
nonlinear_gcnn='relu'
prot_interact_vec_dim='256'
server='/local/datdb'
work_dir=$server/'deepgo/data'
go_enc_model_load=$work_dir/'GCN/'$metric_option'.768'/'best_state_dict.pytorch'
##!! do not have vectors written out for GCN
work_dir=$server/'deepgo/data' ##!! should probably not need this if we don't hardcode extra needed stuffs.
batch_size_aa_go=32
optim_choice='RMSprop'
sgd_lr=0.001
weight_decay=0
for model_choice in DeepGOFlatSeqProtHwayGoNotUsePPI ; do  # DeepGOFlatSeqProtHwayGo

  prot_interact_vec_dim=256
  if [[ $model_choice == 'DeepGOFlatSeqProtHwayGoNotUsePPI' ]]
  then
    prot_interact_vec_dim=0 ##!! so we don't load protein vec
    sgd_lr=0.01
  fi

  if [[ $model_choice == 'DeepGOFlatSeqConcatGo' ]]
  then
    prot_interact_vec_dim=0 ##!! so we don't load protein vec
    sgd_lr=0.0001
  fi

  for lr in 0.001 ; do
    for fold in 1 ; do
      for ontology in mf cc bp ; do # bp mf cc ;

        label_subset_file=$server/'deepgo/data/train/deepgo.'$ontology'.csv'
        label_counter_dict=$server/'deepgo/data/train/fold_1/CountGoInTrain-'$ontology'.pickle'
        deepgo_dir=$server/'deepgo/data/train/fold_'$fold
        data_dir=$deepgo_dir

        # # model_choice='DeepGOFlatSeqProtHwayGo' ##!!
        # model_choice='DeepGOFlatSeqProtHwayGoNotUsePPI'  

        result_folder=$data_dir/$model_choice'GCN'
        mkdir $result_folder
        result_folder=$data_dir/$model_choice'GCN/NotNormalize'
        mkdir $result_folder

        result_folder=$result_folder/$ontology'b'$batch_size_aa_go'lr'$lr$optim_choice # 'SwSgd' ## sub folder
        # result_folder=$result_folder/$ontology'b32lr0.001' # 'SwSgd' ## sub folder
        # prot2seq_model_load=$result_folder/'best_state_dict.pytorch'
        # result_folder=$data_dir/'GCN.'$def_emb_dim ## re-define
        # result_folder=$result_folder/$ontology'b'$batch_size_aa_go'lr'$lr'continue'

        cd $server/GOmultitask/
        mkdir $result_folder
        CUDA_VISIBLE_DEVICES=2 python3 $server/GOmultitask/ProtSeq2GO/do_model_gcn.py --sgd_lr $sgd_lr --model_choice $model_choice --has_ppi_emb --prot_interact_vec_dim $prot_interact_vec_dim --ontology $ontology --do_kmer --gcnn_dim $gcnn_dim --def_emb_dim $def_emb_dim --go_vec_dim $def_emb_dim --prot_vec_dim 832 --optim_choice $optim_choice --lr $lr --main_dir $work_dir --data_dir $data_dir --batch_size_aa_go $batch_size_aa_go --result_folder $result_folder --epoch 80 --use_cuda --metric_option $metric_option --nonlinear_gcnn $nonlinear_gcnn --go_enc_model_load $go_enc_model_load --label_subset_file $label_subset_file --fix_go_emb --switch_sgd > $result_folder/train.log

        # --prot2seq_model_load $prot2seq_model_load --switch_sgd 

        ##!! test
        prot2seq_model_load=$result_folder/'best_state_dict.pytorch'
        CUDA_VISIBLE_DEVICES=2 python3 $server/GOmultitask/ProtSeq2GO/do_model_gcn.py --model_choice $model_choice --has_ppi_emb --prot_interact_vec_dim $prot_interact_vec_dim --ontology $ontology --do_kmer --gcnn_dim $gcnn_dim --def_emb_dim $def_emb_dim --go_vec_dim $def_emb_dim --prot_vec_dim 832 --optim_choice RMSprop --main_dir $work_dir --data_dir $data_dir --batch_size_aa_go 48 --result_folder $result_folder --epoch 0 --use_cuda --metric_option $metric_option --nonlinear_gcnn $nonlinear_gcnn --go_enc_model_load $go_enc_model_load --label_subset_file $label_subset_file --fix_go_emb --prot2seq_model_load $prot2seq_model_load --not_train > $result_folder/test_frequency.log

        #### test zeroshot
        # prot2seq_model_load=$result_folder/'best_state_dict.pytorch' ##!! define again
        # label_subset_file=$server/'deepgo/dataExpandGoSet16Jan2020/train/deepgo.'$ontology'.csv'
        # label_counter_dict=$server/'deepgo/dataExpandGoSet16Jan2020/train/fold_1/CountGoInTrain-'$ontology'.pickle'
        # test_data_name=$server/'deepgo/dataExpandGoSet16Jan2020/train/fold_'$fold/'test-'$ontology'-same-origin.tsv'
        # test_output_name=$result_folder/'test-'$ontology'-same-origin'
        # CUDA_VISIBLE_DEVICES=7 python3 $server/GOmultitask/ProtSeq2GO/do_model_gcn.py --test_data_name $test_data_name --test_output_name $test_output_name --model_choice $model_choice --has_ppi_emb --prot_interact_vec_dim $prot_interact_vec_dim --ontology $ontology --do_kmer --gcnn_dim $gcnn_dim --def_emb_dim $def_emb_dim --go_vec_dim $def_emb_dim --prot_vec_dim 832 --optim_choice RMSprop --main_dir $work_dir --data_dir $data_dir --batch_size_aa_go 64 --result_folder $result_folder --epoch 0 --use_cuda --metric_option $metric_option --nonlinear_gcnn $nonlinear_gcnn --go_enc_model_load $go_enc_model_load --label_subset_file $label_subset_file --fix_go_emb --prot2seq_model_load $prot2seq_model_load --not_train > $result_folder/own_test_frequency.log

      done
    done
  done
done



#### ELMO vector


conda activate tensorflow_gpuenv
def_emb_dim='1024' # COMMENT we're using 1024 in this case, normalized to len=1 ahead of time
metric_option='cosine' ## LOAD BACK COSINE
prot_interact_vec_dim='256'
server='/local/datdb'
work_dir=$server/'deepgo/data'
w2v_emb=$work_dir'/word_pubmed_intersect_GOdb_w2v_pretrained.pickle'
vocab_list=$work_dir/'word_pubmed_intersect_GOdb.txt' # all vocab to be used, can be BERT or something, so we don't hardcode

##!! #### COMMENT vector can be unstable 

precomputed_vector='/local/datdb/deepgo/data/Elmo1024Replace768Unnormalize/label_vector.pickle' ####
work_dir=$server/'deepgo/data' ##!! redefine ... should probably not need this if we don't hardcode extra needed stuffs.
batch_size_aa_go=32 #24 # 32
weight_decay=0
sgd_lr=0.0001
optim_choice='RMSprop'

for model_choice in DeepGOFlatSeqConcatGo DeepGOFlatSeqProtHwayGoNotUsePPI ; do # 'DeepGOFlatSeqProtHwayGo' DeepGOFlatSeqConcatGo

  prot_interact_vec_dim=256
  if [[ $model_choice == 'DeepGOFlatSeqProtHwayGoNotUsePPI' ]]
  then
    prot_interact_vec_dim=0 ##!! so we don't load protein vec
    sgd_lr=0.01
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
        result_folder=$data_dir/$model_choice'ELMONotNormalize' #$add_name
        mkdir $result_folder
        result_folder=$data_dir/$model_choice'ELMONotNormalize/NotNormalize' #$add_name
        mkdir $result_folder

        result_folder=$result_folder/$ontology'b'$batch_size_aa_go'lr'$lr$optim_choice 

        ## COMMENT train
        mkdir $result_folder
        cd $server/GOmultitask/
        CUDA_VISIBLE_DEVICES=2 python3 $server/GOmultitask/ProtSeq2GO/do_model_bilstm.py --sgd_lr $sgd_lr --weight_decay $weight_decay --vocab_list $vocab_list --w2v_emb $w2v_emb --model_choice $model_choice --has_ppi_emb --prot_interact_vec_dim $prot_interact_vec_dim --ontology $ontology --do_kmer --go_vec_dim $def_emb_dim --prot_vec_dim 832 --optim_choice $optim_choice --lr $lr --main_dir $work_dir --data_dir $data_dir --batch_size_aa_go $batch_size_aa_go --result_folder $result_folder --epoch 80 --use_cuda --metric_option $metric_option --precomputed_vector $precomputed_vector --label_subset_file $label_subset_file --fix_go_emb --def_emb_dim $def_emb_dim --reduce_cls_vec --switch_sgd > $result_folder/train.log

        # --prot2seq_model_load $prot2seq_model_load --switch_sgd --seed 2020 ##!! do not need to do this unless forced to

        ## COMMENT testing
        prot2seq_model_load=$result_folder/'best_state_dict.pytorch' ##!! define again
        CUDA_VISIBLE_DEVICES=2 python3 $server/GOmultitask/ProtSeq2GO/do_model_bilstm.py --vocab_list $vocab_list --w2v_emb $w2v_emb --model_choice $model_choice --has_ppi_emb --prot_interact_vec_dim $prot_interact_vec_dim --ontology $ontology --do_kmer --go_vec_dim $def_emb_dim --prot_vec_dim 832 --main_dir $work_dir --data_dir $data_dir --batch_size_aa_go 48 --result_folder $result_folder --use_cuda --metric_option $metric_option --precomputed_vector $precomputed_vector --label_subset_file $label_subset_file --fix_go_emb --def_emb_dim $def_emb_dim --reduce_cls_vec --prot2seq_model_load $prot2seq_model_load --not_train  > $result_folder/test_frequency.log

        #### test zeroshot
        prot2seq_model_load=$result_folder/'best_state_dict.pytorch' ##!! define again
        label_subset_file=$server/'deepgo/dataExpandGoSet16Jan2020/train/deepgo.'$ontology'.csv'
        label_counter_dict=$server/'deepgo/dataExpandGoSet16Jan2020/train/fold_1/CountGoInTrain-'$ontology'.pickle'
        test_data_name=$server/'deepgo/dataExpandGoSet16Jan2020/train/fold_'$fold/'test-'$ontology'-same-origin.tsv'
        test_output_name=$result_folder/'test-'$ontology'-same-origin'
        CUDA_VISIBLE_DEVICES=2 python3 $server/GOmultitask/ProtSeq2GO/do_model_bilstm.py --test_data_name $test_data_name --test_output_name $test_output_name --vocab_list $vocab_list --w2v_emb $w2v_emb --model_choice $model_choice --has_ppi_emb --prot_interact_vec_dim $prot_interact_vec_dim --ontology $ontology --do_kmer --go_vec_dim $def_emb_dim --prot_vec_dim 832 --main_dir $work_dir --data_dir $data_dir --batch_size_aa_go 48 --result_folder $result_folder --use_cuda --metric_option $metric_option --precomputed_vector $precomputed_vector --label_subset_file $label_subset_file --fix_go_emb --def_emb_dim $def_emb_dim --reduce_cls_vec --prot2seq_model_load $prot2seq_model_load --not_train > $result_folder/own_test_frequency.log

      done
    done
  done
done



