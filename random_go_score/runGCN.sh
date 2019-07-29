

server='/local/datdb'

## use cosine similarity as objective function 
def_emb_dim='300'
metric_option='cosine'
nonlinear_gcnn='relu'

server='/local/datdb'
work_dir=$server/'goAndGeneAnnotationMar2017'

data_dir=$server/'goAndGeneAnnotationMar2017/RandomGOAnalysis'
mkdir $work_dir/'RandomGOAnalysis/GCN'

model_load=$work_dir/'GCN/'$metric_option'.300/best_state_dict.pytorch'

result_folder=$work_dir/'RandomGOAnalysis/GCN/'$metric_option'.300' #$def_emb_dim.'clsVec'
mkdir $result_folder


for file_type in random_go_analysis_cc random_go_analysis_bp ParentChild_go_analysis_cc ParentChild_go_analysis_bp ; do 

  test_file=$work_dir/$file_type'.tsv'
  write_score=$result_folder/$file_type'.GCN.temp.txt'

  ## set epoch=0 for testing
  conda activate tensorflow_gpuenv
  cd $server/GOmultitask

  CUDA_VISIBLE_DEVICES=1 python3 $server/GOmultitask/GCN/encoder/do_model.py --main_dir $work_dir --qnli_dir $data_dir --batch_size_label 128 --result_folder $result_folder --epoch 0 --use_cuda --metric_option $metric_option --nonlinear_gcnn $nonlinear_gcnn --def_emb_dim $def_emb_dim --model_load $model_load --test_file $test_file --write_score $write_score > $result_folder/test2.log

  paste $test_file $write_score > $result_folder/$file_type'.GCN.txt' ## append columns 

done 



##  see GO:1903047 having a lot of children terms and also very bad score
