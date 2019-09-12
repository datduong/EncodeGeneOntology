
## run on deepgo, ADD our GO encoder, ADD IN PROT-PROT EMB + sequence + flat (no maxpool over child nodes), this requires 3 runs, bp, mf, cc 
## run fast with if we don't update go definitions embedding 

conda activate tensorflow_gpuenv

def_emb_dim='768'
metric_option='cosine' ## LOAD BACK COSINE
prot_interact_vec_dim='256'

server='/local/datdb'
work_dir=$server/'deepgo/data'
w2v_emb=$work_dir'/word_pubmed_intersect_GOdb_w2v_pretrained.pickle'
vocab_list=$server/'deepgo/data/word_pubmed_intersect_GOdb.txt' 

precomputed_vector='/local/datdb/Onto2Vec/GOVectorData/2016DeepGOData/onto2vec2016dim768.pickle'

work_dir=$server/'deepgo/dataExpandGoSet' ## should probably not need this if we don't hardcode extra needed stuffs.
for fold in 1 ; do 
  for ontology in mf cc bp ; do # cc bp bp mf cc

    label_subset_file=$server/'deepgo/dataExpandGoSet/train/deepgo.'$ontology'.csv'
    label_counter_dict=$server/'deepgo/dataExpandGoSet/train/CountGoInTrain-'$ontology'.pickle'
    deepgo_dir=$server/'deepgo/dataExpandGoSet/train/fold_'$fold
    data_dir=$deepgo_dir

    model_choice='DeepGOFlatSeqProtHwayGo'
    result_folder=$data_dir/'FlatSeqProtHwayOnto2vecLr0.001b64.'$def_emb_dim.$ontology
    # prot2seq_model_load=$result_folder/'best_state_dict.pytorch'

    mkdir $result_folder

    cd $server/GOmultitask/

    ## 0.0001 24
    ## 0.001 64

    CUDA_VISIBLE_DEVICES=2 python3 $server/GOmultitask/ProtSeq2GO/do_model_bilstm.py --bilstm_dim 1024 --vocab_list $vocab_list --w2v_emb $w2v_emb --model_choice $model_choice --has_ppi_emb --prot_interact_vec_dim $prot_interact_vec_dim --ontology $ontology --do_kmer --go_vec_dim $def_emb_dim --prot_vec_dim 832 --optim_choice RMSprop --lr 0.001 --main_dir $work_dir --data_dir $data_dir --batch_size_label 64 --result_folder $result_folder --epoch 50 --use_cuda --metric_option $metric_option --precomputed_vector $precomputed_vector --label_subset_file $label_subset_file --fix_go_emb --def_emb_dim $def_emb_dim --reduce_cls_vec > $result_folder/train.log # --prot2seq_model_load $prot2seq_model_load

    prot2seq_model_load=$result_folder/'best_state_dict.pytorch'

    ## for testing
    CUDA_VISIBLE_DEVICES=2 python3 $server/GOmultitask/ProtSeq2GO/do_model_bilstm.py --bilstm_dim 1024 --vocab_list $vocab_list --w2v_emb $w2v_emb --model_choice $model_choice --has_ppi_emb --prot_interact_vec_dim $prot_interact_vec_dim --ontology $ontology --do_kmer --go_vec_dim $def_emb_dim --prot_vec_dim 832 --main_dir $work_dir --data_dir $data_dir --batch_size_label 128 --result_folder $result_folder --epoch 50 --use_cuda --metric_option $metric_option --precomputed_vector $precomputed_vector --label_subset_file $label_subset_file --fix_go_emb --def_emb_dim $def_emb_dim --reduce_cls_vec --prot2seq_model_load $prot2seq_model_load --not_train --label_counter_dict $label_counter_dict > $result_folder/test_frequency.log # --prot2seq_model_load $prot2seq_model_load

  done
done 

