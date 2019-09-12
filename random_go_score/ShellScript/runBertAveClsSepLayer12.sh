




server='/local/datdb'

## use cosine similarity as objective function 
def_emb_dim='768'
metric_option='cosine'

work_dir=$server/'goAndGeneAnnotationMar2017'
bert_model=$work_dir/'BERT_base_cased_tune_go_branch/fine_tune_lm_bioBERT' # use the full mask + nextSentence to innit
data_dir=$server/'goAndGeneAnnotationMar2017/entailment_data/AicScore/go_bert_cls'
pregenerated_data=$server/'goAndGeneAnnotationMar2017/BERT_base_cased_tune_go_branch' # use the data of full mask + nextSentence to innit
bert_output_dir=$pregenerated_data/'fine_tune_lm_bioBERT'

result_folder=$bert_output_dir/'cosine.AveWordClsSep768.Linear768.Layer12' 
model_load=$result_folder/'best_state_dict.pytorch'

## redefine the @result_folder 
mkdir $work_dir/'RandomGOAnalysis/BERT/'
result_folder=$work_dir/'RandomGOAnalysis/BERT/cosine.AveWordClsSep768.Linear768.Layer12'
mkdir $result_folder

conda activate tensorflow_gpuenv
cd $server/GOmultitask

# random_go_analysis_cc random_go_analysis_bp ParentChild_go_analysis_mf ParentChild_go_analysis_cc ParentChild_go_analysis_bp

for file_type in random_go_analysis_mf random_go_analysis_cc random_go_analysis_bp ParentChild_go_analysis_mf ParentChild_go_analysis_cc ParentChild_go_analysis_bp ; do 

  test_file=$work_dir/$file_type'.tsv'
  write_score=$result_folder/$file_type'.BERT.temp.txt'

  ## set epoch=0 for testing

  ## check the Average-layer if not use CLS 
  
  CUDA_VISIBLE_DEVICES=5 python3 $server/GOmultitask/BERT/encoder/do_model.py --main_dir $work_dir --qnli_dir $data_dir --batch_size_label 64 --bert_model $bert_model --pregenerated_data $pregenerated_data --bert_output_dir $bert_output_dir --result_folder $result_folder --epoch 0 --num_train_epochs_entailment 0 --use_cuda --metric_option $metric_option --def_emb_dim $def_emb_dim --reduce_cls_vec --model_load $model_load --write_score $write_score --test_file $test_file --average_layer --layer_index 1 > $result_folder/test1.log

  paste $test_file $write_score > $result_folder/$file_type'.BERT.txt' ## append columns 
  rm -f $write_score

done






