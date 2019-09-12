

server='/local/datdb'

## use cosine similarity as objective function 
def_emb_dim='768'
metric_option='cosine'
nonlinear_gcnn='relu'

server='/local/datdb'
work_dir=$server/'goAndGeneAnnotationMar2017'
model_load=$work_dir/'cosine.bilstm768/best_state_dict.pytorch'

data_dir=$server/'goAndGeneAnnotationMar2017/RandomGOAnalysis'
mkdir $work_dir/'RandomGOAnalysis/BiLSTM'
result_folder=$work_dir/'RandomGOAnalysis/BiLSTM/cosine.bilstm768'
mkdir $result_folder

w2v_emb=$work_dir'/word_pubmed_intersect_GOdb_w2v_pretrained.pickle'
vocab_list='word_pubmed_intersect_GOdb.txt' # all vocab to be used, can be BERT or something, so we don't hardcode 

for file_type in random_go_analysis_mf random_go_analysis_cc random_go_analysis_bp ParentChild_go_analysis_mf ParentChild_go_analysis_cc ParentChild_go_analysis_bp ; do 

  test_file=$work_dir/$file_type'.tsv'
  write_score=$result_folder/$file_type'.BiLSTM.temp.txt'

  ## set epoch=0 for testing
  conda activate tensorflow_gpuenv
  cd $server/GOmultitask

  CUDA_VISIBLE_DEVICES=5 python3 $server/GOmultitask/biLSTM/encoder/do_model.py --model_load $model_load --fix_word_emb --vocab_list $vocab_list --w2v_emb $w2v_emb --bilstm_dim 1024 --main_dir $work_dir --qnli_dir $data_dir --batch_size_label 128 --result_folder $result_folder --epoch 0 --use_cuda --metric_option $metric_option --def_emb_dim $def_emb_dim --reduce_cls_vec --test_file $test_file --write_score $write_score > $result_folder/test1.log

  paste $test_file $write_score > $result_folder/$file_type'.BiLSTM.txt' ## append columns 

done 



##  see GO:1903047 having a lot of children terms and also very bad score
