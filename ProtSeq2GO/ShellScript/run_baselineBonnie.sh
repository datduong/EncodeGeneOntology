
## run on deepgo, ADD our GO encoder, ADD IN PROT-PROT EMB + sequence + flat (no maxpool over child nodes), this requires 3 runs, bp, mf, cc 
## run fast with if we don't update go definitions embedding 

conda activate tensorflow_gpuenv

def_emb_dim='300' ## doesn't matter, just dummy, baseline doesn't use go vector 
metric_option='cosine' ## LOAD BACK COSINE
prot_interact_vec_dim='100' ## COMMENT

server='/local/datdb'
work_dir=$server/'deepgo/data' ##!! will be re-defined later

w2v_emb=$work_dir'/word_pubmed_intersect_GOdb_w2v_pretrained.pickle'
vocab_list=$server/'deepgo/data/word_pubmed_intersect_GOdb.txt'

work_dir=$server/'deepgo/dataExpandGoSet' ## should probably not need this if we don't hardcode extra needed stuffs.
batch_size_label=32 #24 # 32
# lr=0.001
for lr in 0.001 ; do
  for fold in 1 ; do 
    for ontology in bp cc ; do # cc bp mf

    label_in_ontology=$work_dir/'terms_in_'$ontology'.csv'
    label_subset_file=$server/'deepgo/dataExpandGoSet/train/deepgo.'$ontology'.csv'

    deepgo_dir=$server/'deepgo/dataExpandGoSet/train/fold_'$fold
    data_dir=$deepgo_dir

    # model_choice='DeepGOFlatSeqProt' #'DeepGOFlatSeqProtHwayNotUseGo'
    model_choice='DeepGOFlatSeqProtHwayNotUseGo' ####
    result_folder=$data_dir/$model_choice'Bonnie'
    mkdir $result_folder
    result_folder=$result_folder/$ontology'b'$batch_size_label'lr'$lr
    mkdir $result_folder

    # prot2seq_model_load=$result_folder/'last_state_dict.pytorch'
    label_counter_dict=$deepgo_dir/'CountGoInTrain-'$ontology'.pickle'

    cd $server/GOmultitask/

    ##!! calling @do_model_bilstm doesn't matter, because it will just do whatever the model @model_choice does 

    CUDA_VISIBLE_DEVICES=5 python3 $server/GOmultitask/ProtSeq2GO/do_model_bilstm.py --add_name same-origin-bonnie --bilstm_dim 1024 --vocab_list $vocab_list --w2v_emb $w2v_emb --model_choice $model_choice --has_ppi_emb --prot_interact_vec_dim $prot_interact_vec_dim --ontology $ontology --do_kmer --go_vec_dim 0 --prot_vec_dim 832 --optim_choice RMSprop --lr $lr --main_dir $work_dir --data_dir $data_dir --batch_size_label $batch_size_label --result_folder $result_folder --epoch 50 --use_cuda --metric_option $metric_option --label_subset_file $label_subset_file --label_in_ontology $label_in_ontology --fix_go_emb --def_emb_dim $def_emb_dim --reduce_cls_vec --switch_sgd > $result_folder/train.log ## --prot2seq_model_load $prot2seq_model_load --go_enc_model_load $go_enc_model_load 

    #### test
    prot2seq_model_load=$result_folder/'best_state_dict.pytorch'
    CUDA_VISIBLE_DEVICES=5 python3 $server/GOmultitask/ProtSeq2GO/do_model_bilstm.py --add_name same-origin-bonnie --bilstm_dim 1024 --vocab_list $vocab_list --w2v_emb $w2v_emb --model_choice $model_choice --has_ppi_emb --prot_interact_vec_dim $prot_interact_vec_dim --ontology $ontology --do_kmer --go_vec_dim 0 --prot_vec_dim 832 --main_dir $work_dir --data_dir $data_dir --batch_size_label 64 --result_folder $result_folder --use_cuda --metric_option $metric_option --label_subset_file $label_subset_file --label_in_ontology $label_in_ontology --fix_go_emb --def_emb_dim $def_emb_dim --reduce_cls_vec --prot2seq_model_load $prot2seq_model_load --not_train --label_counter_dict $label_counter_dict > $result_folder/test_frequency.log # --prot2seq_model_load $prot2seq_model_load

    done
  done
done
