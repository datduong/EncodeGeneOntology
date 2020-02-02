

conda activate tensorflow_gpuenv

def_emb_dim='300' ## doesn't matter, just dummy, baseline doesn't use go vector
metric_option='cosine' ## LOAD BACK COSINE
prot_interact_vec_dim='100' ## COMMENT

server='/local/datdb'
work_dir=$server/'deepgo/data' ##!! will be re-defined later

w2v_emb=$work_dir'/word_pubmed_intersect_GOdb_w2v_pretrained.pickle'
vocab_list=$server/'deepgo/data/word_pubmed_intersect_GOdb.txt'

work_dir=$server/'deepgo/data' ## should probably not need this if we don't hardcode extra needed stuffs.
batch_size_aa_go=32
sgd_lr=0.05
optim_choice='RMSprop'

for lr in 0.001 ; do
  for fold in 1 ; do
    for ontology in mf bp cc ; do # cc bp mf

    label_in_ontology=$work_dir/'terms_in_'$ontology'.csv'
    label_subset_file=$server/'deepgo/data/train/deepgo.'$ontology'.csv'

    deepgo_dir=$server/'deepgo/data/train/fold_'$fold
    data_dir=$deepgo_dir

    model_choice='DeepGOFlatSeqProt' ####
    result_folder=$data_dir/$model_choice'Bonnie'
    mkdir $result_folder
    result_folder=$result_folder/$ontology'b'$batch_size_aa_go'lr'$lr
    label_counter_dict=$deepgo_dir/'CountGoInTrain-'$ontology'.pickle'

    cd $server/GOmultitask/
    mkdir $result_folder

    if [[ $model_choice == 'DeepGOFlatSeqProtHwayNotUseGo' ]]
    then
    def_emb_dim=0 ##!! so we don't load protein vec
    fi

    if [[ $model_choice == 'DeepGOFlatSeqProt' ]]
    then
    def_emb_dim=0 ##!! so we don't load protein vec
    fi

    ##!! calling @do_model_bilstm doesn't matter, because it will just do whatever the model @model_choice does

    CUDA_VISIBLE_DEVICES=6 python3 $server/GOmultitask/ProtSeq2GO/do_model_bilstm.py --add_name bonnie --sgd_lr $sgd_lr --bilstm_dim 1024 --vocab_list $vocab_list --w2v_emb $w2v_emb --model_choice $model_choice --has_ppi_emb --prot_interact_vec_dim $prot_interact_vec_dim --ontology $ontology --do_kmer --go_vec_dim 0 --prot_vec_dim 832 --optim_choice $optim_choice --lr $lr --main_dir $work_dir --data_dir $data_dir --batch_size_aa_go $batch_size_aa_go --result_folder $result_folder --epoch 50 --use_cuda --metric_option $metric_option --label_subset_file $label_subset_file --label_in_ontology $label_in_ontology --fix_go_emb --def_emb_dim $def_emb_dim --reduce_cls_vec --switch_sgd > $result_folder/train.log

    ## --prot2seq_model_load $prot2seq_model_load --go_enc_model_load $go_enc_model_load

    #### test
    prot2seq_model_load=$result_folder/'best_state_dict.pytorch'
    CUDA_VISIBLE_DEVICES=6 python3 $server/GOmultitask/ProtSeq2GO/do_model_bilstm.py --add_name bonnie --bilstm_dim 1024 --vocab_list $vocab_list --w2v_emb $w2v_emb --model_choice $model_choice --has_ppi_emb --prot_interact_vec_dim $prot_interact_vec_dim --ontology $ontology --do_kmer --go_vec_dim 0 --prot_vec_dim 832 --main_dir $work_dir --data_dir $data_dir --batch_size_aa_go 64 --result_folder $result_folder --use_cuda --metric_option $metric_option --label_subset_file $label_subset_file --label_in_ontology $label_in_ontology --fix_go_emb --def_emb_dim $def_emb_dim --reduce_cls_vec --prot2seq_model_load $prot2seq_model_load --not_train --label_counter_dict $label_counter_dict > $result_folder/test_frequency.log

    # --prot2seq_model_load $prot2seq_model_load

    done
  done
done



