
conda activate tensorflow_gpuenv

def_emb_dim='300'
metric_option='cosine' ## LOAD BACK COSINE
nonlinear_gcnn='relu'
prot_interact_vec_dim='256'

server='/local/datdb'
# work_dir=$server/'goAndGeneAnnotationDec2018'
work_dir=$server/'deepgo/data'

go_enc_model_load=$work_dir/'GCN/'$metric_option'.300'/'best_state_dict.pytorch'

for fold in 1 ; do 

  for ontology in mf cc bp ; do

    label_subset_file=$server/'deepgo/data/train/deepgo.'$ontology'.csv'

    deepgo_dir=$server/'deepgo/data/train/fold_'$fold

    data_dir=$deepgo_dir

    model_choice='DeepGOFlatSeqProtHwayGo'
    result_folder=$data_dir/'FlatSeqProtHwayGCNGoRun2.'$ontology
    # mkdir $result_folder
    prot2seq_model_load=$result_folder/'last_state_dict.pytorch'
    label_counter_dict=$deepgo_dir/'CountGoInTrain-'$ontology'.pickle'

    cd $server/GOmultitask/

    CUDA_VISIBLE_DEVICES=6 python3 $server/GOmultitask/ProtSeq2GO/do_model_gcn.py --model_choice $model_choice --has_ppi_emb --prot_interact_vec_dim $prot_interact_vec_dim --ontology $ontology --do_kmer --go_vec_dim 300 --prot_vec_dim 832 --optim_choice RMSprop --lr 0.0001 --main_dir $work_dir --data_dir $data_dir --batch_size_label 24 --result_folder $result_folder --epoch 0 --use_cuda --metric_option $metric_option --nonlinear_gcnn $nonlinear_gcnn --go_enc_model_load $go_enc_model_load --label_subset_file $label_subset_file --fix_go_emb --prot2seq_model_load $prot2seq_model_load --not_train --label_counter_dict $label_counter_dict > $result_folder/test_frequency.log

  done

done 

