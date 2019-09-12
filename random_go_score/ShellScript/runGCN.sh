

server='/local/datdb'
conda activate tensorflow_gpuenv
## use cosine similarity as objective function 
def_emb_dim='768'
gcnn_dim='768'
metric_option='cosine'
nonlinear_gcnn='relu'

server='/local/datdb'
work_dir=$server/'goAndGeneAnnotationMar2017'

data_dir=$server/'goAndGeneAnnotationMar2017/RandomGOAnalysis'
mkdir $work_dir/'RandomGOAnalysis/GCN'

model_load=$work_dir/'GCN/'$metric_option'.768/best_state_dict.pytorch'

result_folder=$work_dir/'RandomGOAnalysis/GCN/'$metric_option'.768' #$def_emb_dim.'clsVec'
mkdir $result_folder

# word_mode='PretrainedGO'
# w2v_emb='/local/datdb/Onto2Vec/GOVectorData/2017/onto2vec_embeddings.pickle' ## use 2017 data here. 

for file_type in random_go_analysis_mf ParentChild_go_analysis_mf random_go_analysis_cc random_go_analysis_bp ParentChild_go_analysis_cc ParentChild_go_analysis_bp ; do 

  test_file=$work_dir/$file_type'.tsv'
  write_score=$result_folder/$file_type'.GCN.temp.txt'

  ## set epoch=0 for testing
  conda activate tensorflow_gpuenv
  cd $server/GOmultitask

  CUDA_VISIBLE_DEVICES=5 python3 $server/GOmultitask/GCN/encoder/do_model.py --main_dir $work_dir --qnli_dir $data_dir --batch_size_label 256 --result_folder $result_folder --epoch 0 --use_cuda --metric_option $metric_option --nonlinear_gcnn $nonlinear_gcnn --def_emb_dim $def_emb_dim --model_load $model_load --test_file $test_file --write_score $write_score --gcnn_dim $gcnn_dim > $result_folder/test2.log

  # --w2v_emb $w2v_emb --word_mode $word_mode

  paste $test_file $write_score > $result_folder/$file_type'.GCNOnto2vec.txt' ## append columns 

done 



##  see GO:1903047 having a lot of children terms and also very bad score
