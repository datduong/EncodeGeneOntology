
## run on deepgo, ADD our GO encoder, ADD IN PROT-PROT EMB + sequence + flat (no maxpool over child nodes), this requires 3 runs, bp, mf, cc 
## run fast with if we don't update go definitions embedding 

## we will write the BERT vector in a pickle, and then we load it. 
## we can reuse the same code to run a "precomputed vector"

conda activate tensorflow_gpuenv
def_emb_dim='768'
metric_option='cosine' ## LOAD BACK COSINE
prot_interact_vec_dim='256'

server='/local/datdb'

## dont need to move these files over 
w2v_emb=$server/'deepgo/data/word_pubmed_intersect_GOdb_w2v_pretrained.pickle'
vocab_list=$server/'deepgo/data/word_pubmed_intersect_GOdb.txt' # all vocab to be used, can be BERT or something, so we don't hardcode 
precomputed_vector='/local/datdb/deepgo/data/cosine.AveWordClsSep768.Linear768.Layer12/label_vector.pickle'


work_dir=$server/'deepgo/dataExpandGoSet' ## should probably not need this if we don't hardcode extra needed stuffs.
for fold in 1 ; do 
  for ontology in cc ; do # cc bp bp mf cc

    label_subset_file=$server/'deepgo/dataExpandGoSet/train/deepgo.'$ontology'.csv'
    label_counter_dict=$server/'deepgo/dataExpandGoSet/train/CountGoInTrain-'$ontology'.pickle'
    deepgo_dir=$server/'deepgo/dataExpandGoSet/train/fold_'$fold
    data_dir=$deepgo_dir
    model_choice='DeepGOFlatSeqProtHwayGo'
    result_folder=$data_dir/'BertAveWordClsSepRmsL12Lr0.001b128.'$def_emb_dim.$ontology
    prot2seq_model_load=$result_folder/'best_state_dict.pytorch'

    ## continue training redefine result, and save to new location
    # result_folder=$data_dir/'FlatSeqProtHwayBertAveWordClsSepL12PretrainR2Lr0.001b32.'$def_emb_dim.$ontology
    mkdir $result_folder

    cd $server/GOmultitask/ 

    ## 0.0001 24
    ## 0.001 64

    ## we will write the BERT vector in a pickle, and then we load it. 
    ## we can reuse the same code to run a "precomputed vector"
    # --label_in_ontology $label_in_ontology 
    CUDA_VISIBLE_DEVICES=0 python3 $server/GOmultitask/ProtSeq2GO/do_model_bilstm.py --bilstm_dim 1024 --vocab_list $vocab_list --w2v_emb $w2v_emb --model_choice $model_choice --has_ppi_emb --prot_interact_vec_dim $prot_interact_vec_dim --ontology $ontology --do_kmer --go_vec_dim $def_emb_dim --prot_vec_dim 832 --optim_choice RMSprop --lr 0.001 --main_dir $work_dir --data_dir $data_dir --batch_size_label 128 --result_folder $result_folder --epoch 20 --use_cuda --metric_option $metric_option --precomputed_vector $precomputed_vector --label_subset_file $label_subset_file --fix_go_emb --def_emb_dim $def_emb_dim --reduce_cls_vec > $result_folder/train2.log # --prot2seq_model_load $prot2seq_model_load

    prot2seq_model_load=$result_folder/'best_state_dict.pytorch'

    ## for testing
    CUDA_VISIBLE_DEVICES=0 python3 $server/GOmultitask/ProtSeq2GO/do_model_bilstm.py --bilstm_dim 1024 --vocab_list $vocab_list --w2v_emb $w2v_emb --model_choice $model_choice --has_ppi_emb --prot_interact_vec_dim $prot_interact_vec_dim --ontology $ontology --do_kmer --go_vec_dim $def_emb_dim --prot_vec_dim 832 --optim_choice RMSprop --lr 0.001 --main_dir $work_dir --data_dir $data_dir --batch_size_label 64 --result_folder $result_folder --use_cuda --metric_option $metric_option --precomputed_vector $precomputed_vector --label_subset_file $label_subset_file --fix_go_emb --def_emb_dim $def_emb_dim --reduce_cls_vec --prot2seq_model_load $prot2seq_model_load --not_train --label_counter_dict $label_counter_dict > $result_folder/test_frequency2.log # --prot2seq_model_load $prot2seq_model_load

  done
done 

