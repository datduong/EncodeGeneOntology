
## run on deepgo, ADD our GO encoder, ADD IN PROT-PROT EMB + sequence + flat (no maxpool over child nodes), this requires 3 runs, bp, mf, cc
## run fast with if we don't update go definitions embedding

## we will write the BERT vector in a pickle, and then we load it.
## we can reuse the same code to run a "precomputed vector"

conda activate tensorflow_gpuenv
def_emb_dim='768'
metric_option='cosine' ## LOAD BACK COSINE
prot_interact_vec_dim='256'

server='/local/datdb'
work_dir=$server/'deepgo/data'
w2v_emb=$work_dir'/word_pubmed_intersect_GOdb_w2v_pretrained.pickle'
vocab_list=$work_dir/'word_pubmed_intersect_GOdb.txt' # all vocab to be used, can be BERT or something, so we don't hardcode

# go_enc_model_load=$work_dir/$metric_option'.bilstm.300Vec/best_state_dict.pytorch'
precomputed_vector='/local/datdb/deepgo/data/cosine768Linear768Layer11+12nnParamOpW2/label_vector.pickle'


work_dir=$server/'deepgo/dataExpandGoSet' #### redefine ... should probably not need this if we don't hardcode extra needed stuffs.
batch_size_label=32 #24 # 32
weight_decay=0
for lr in 0.001 ; do
  for fold in 1 ; do
    for ontology in mf cc bp ; do # cc bp bp mf cc

      # if [[ $ontology == 'bp' ]]
      # then
      #   batch_size_label=32 #32
      # fi

      label_subset_file=$server/'deepgo/dataExpandGoSet/train/deepgo.'$ontology'.csv'
      label_counter_dict=$server/'deepgo/dataExpandGoSet/train/CountGoInTrain-'$ontology'.pickle'
      deepgo_dir=$server/'deepgo/dataExpandGoSet/train/fold_'$fold
      data_dir=$deepgo_dir

      # model_choice='DeepGOFlatSeqProtHwayGo' ## COMMENT, swap with the other models if needed
      model_choice='DeepGOFlatSeqProtHwayGoNotUsePPI'

      result_folder=$data_dir/$model_choice'Bertd11d12.'$def_emb_dim
      mkdir $result_folder

      result_folder=$result_folder/$ontology'b'$batch_size_label'lr'$lr #'d'$weight_decay # 'SwSgd' ## sub folder
      # result_folder=$result_folder/$ontology'b32lr0.001' # 'SwSgd' ## sub folder
      # prot2seq_model_load=$result_folder/'best_state_dict.pytorch'
      # result_folder=$data_dir/'BertCosLinear768Layer11+12OpW2.'$def_emb_dim ## re-define
      # result_folder=$result_folder/$ontology'b'$batch_size_label'lr'$lr'continue'

      ## COMMENT train
      mkdir $result_folder
      cd $server/GOmultitask/
      CUDA_VISIBLE_DEVICES=2 python3 $server/GOmultitask/ProtSeq2GO/do_model_bilstm.py --weight_decay $weight_decay --add_name same-origin --vocab_list $vocab_list --w2v_emb $w2v_emb --model_choice $model_choice --has_ppi_emb --prot_interact_vec_dim $prot_interact_vec_dim --ontology $ontology --do_kmer --go_vec_dim $def_emb_dim --prot_vec_dim 832 --optim_choice RMSprop --lr $lr --main_dir $work_dir --data_dir $data_dir --batch_size_label $batch_size_label --result_folder $result_folder --epoch 100 --use_cuda --metric_option $metric_option --precomputed_vector $precomputed_vector --label_subset_file $label_subset_file --fix_go_emb --def_emb_dim $def_emb_dim --reduce_cls_vec --switch_sgd > $result_folder/train.log

      # --prot2seq_model_load $prot2seq_model_load --switch_sgd --seed 2020 ##!! do not need to do this unless forced to

      ## COMMENT testing
      prot2seq_model_load=$result_folder/'best_state_dict.pytorch' ##!! define again
      CUDA_VISIBLE_DEVICES=2 python3 $server/GOmultitask/ProtSeq2GO/do_model_bilstm.py --add_name same-origin --vocab_list $vocab_list --w2v_emb $w2v_emb --model_choice $model_choice --has_ppi_emb --prot_interact_vec_dim $prot_interact_vec_dim --ontology $ontology --do_kmer --go_vec_dim $def_emb_dim --prot_vec_dim 832 --main_dir $work_dir --data_dir $data_dir --batch_size_label 128 --result_folder $result_folder --use_cuda --metric_option $metric_option --precomputed_vector $precomputed_vector --label_subset_file $label_subset_file --fix_go_emb --def_emb_dim $def_emb_dim --reduce_cls_vec --prot2seq_model_load $prot2seq_model_load --not_train --label_counter_dict $label_counter_dict > $result_folder/test_frequency.log

    done
  done
done

