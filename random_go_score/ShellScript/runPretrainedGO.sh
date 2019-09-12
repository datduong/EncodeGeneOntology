



server='/local/datdb'
conda activate tensorflow_gpuenv
## use cosine similarity as objective function 
def_emb_dim='768'
metric_option='cosine'
nonlinear_gcnn='relu'

server='/local/datdb'
work_dir=$server/'goAndGeneAnnotationMar2017'

data_dir=$server/'goAndGeneAnnotationMar2017/RandomGOAnalysis'
mkdir $work_dir/'RandomGOAnalysis/Onto2vec'
model_load=$work_dir/'Onto2vec/'$metric_option'.'$def_emb_dim'/best_state_dict.pytorch' ## not used, because vectors are pretrained 
result_folder=$work_dir/'RandomGOAnalysis/Onto2vec/'$metric_option'.'$def_emb_dim
mkdir $result_folder

vector_file='/local/datdb/Onto2Vec/GOVectorData/2017/onto2vec2017dim768.pickle'
vocab_list='word_pubmed_intersect_GOdb.txt' # all vocab to be used, can be BERT or something, so we don't hardcode 

for file_type in random_go_analysis_mf ParentChild_go_analysis_mf random_go_analysis_cc random_go_analysis_bp ParentChild_go_analysis_cc ParentChild_go_analysis_bp ; do 

  test_file=$work_dir/$file_type'.tsv'
  write_score=$result_folder/$file_type'.Onto2vec.temp.txt'

  ## set epoch=0 for testing
  cd $server/GOmultitask

  CUDA_VISIBLE_DEVICES=5 python3 $server/GOmultitask/Precomputed_vector/encoder/do_model.py --main_dir $work_dir --qnli_dir $data_dir --batch_size_label 256 --result_folder $result_folder --epoch 0 --use_cuda --metric_option $metric_option --nonlinear_gcnn $nonlinear_gcnn --def_emb_dim $def_emb_dim --model_load $model_load --test_file $test_file --write_score $write_score --vocab_list $vocab_list --vector_file $vector_file > $result_folder/test2.log

  paste $test_file $write_score > $result_folder/$file_type'.Onto2vec.txt' ## append columns 

done 



