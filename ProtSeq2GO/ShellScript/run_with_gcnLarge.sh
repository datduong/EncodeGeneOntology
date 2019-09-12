

conda activate tensorflow_gpuenv

def_emb_dim='768'
gcnn_dim=$def_emb_dim
metric_option='cosine' ## LOAD BACK COSINE
nonlinear_gcnn='relu'
prot_interact_vec_dim='256'

server='/local/datdb'
work_dir=$server/'deepgo/data'
go_enc_model_load=$work_dir/'GCN/'$metric_option'.768'/'best_state_dict.pytorch'

work_dir=$server/'deepgo/dataExpandGoSet' ## should probably not need this if we don't hardcode extra needed stuffs.
for fold in 1 ; do 

  for ontology in cc ; do # bp mf cc ;

    label_subset_file=$server/'deepgo/dataExpandGoSet/train/deepgo.'$ontology'.csv'
    label_counter_dict=$server/'deepgo/dataExpandGoSet/train/CountGoInTrain-'$ontology'.pickle'
    deepgo_dir=$server/'deepgo/dataExpandGoSet/train/fold_'$fold
    data_dir=$deepgo_dir

    model_choice='DeepGOFlatSeqProtHwayGo'
    result_folder=$data_dir/'FlatSeqProtHwayGCN768Lr0.001b128.'$ontology
    mkdir $result_folder
   
    cd $server/GOmultitask/

    # prot2seq_model_load=$result_folder/'best_state_dict.pytorch'

    CUDA_VISIBLE_DEVICES=5 python3 $server/GOmultitask/ProtSeq2GO/do_model_gcn.py --model_choice $model_choice --has_ppi_emb --prot_interact_vec_dim $prot_interact_vec_dim --ontology $ontology --do_kmer --gcnn_dim $gcnn_dim --def_emb_dim $def_emb_dim --go_vec_dim $def_emb_dim --prot_vec_dim 832 --optim_choice RMSprop --lr 0.001 --main_dir $work_dir --data_dir $data_dir --batch_size_label 128 --result_folder $result_folder --epoch 20 --use_cuda --metric_option $metric_option --nonlinear_gcnn $nonlinear_gcnn --go_enc_model_load $go_enc_model_load --label_subset_file $label_subset_file --fix_go_emb --label_counter_dict $label_counter_dict > $result_folder/train.log

    # --w2v_emb $w2v_emb RMSprop --prot2seq_model_load $prot2seq_model_load

    prot2seq_model_load=$result_folder/'best_state_dict.pytorch'

    CUDA_VISIBLE_DEVICES=5 python3 $server/GOmultitask/ProtSeq2GO/do_model_gcn.py --model_choice $model_choice --has_ppi_emb --prot_interact_vec_dim $prot_interact_vec_dim --ontology $ontology --do_kmer --gcnn_dim $gcnn_dim --def_emb_dim $def_emb_dim --go_vec_dim $def_emb_dim --prot_vec_dim 832 --optim_choice RMSprop --lr 0.0001 --main_dir $work_dir --data_dir $data_dir --batch_size_label 64 --result_folder $result_folder --epoch 0 --use_cuda --metric_option $metric_option --nonlinear_gcnn $nonlinear_gcnn --go_enc_model_load $go_enc_model_load --label_subset_file $label_subset_file --fix_go_emb --prot2seq_model_load $prot2seq_model_load --not_train --label_counter_dict $label_counter_dict > $result_folder/test_frequency.log

    # --word_mode $word_mode --w2v_emb $w2v_emb

  done

done 


