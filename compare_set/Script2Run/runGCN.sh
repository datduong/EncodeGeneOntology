

def_emb_dim='768'
gcnn_dim='768'
metric_option='cosine' ## use cosine similarity as objective function 
nonlinear_gcnn='relu'

server='/local/datdb'
work_dir=$server/'goAndGeneAnnotationMar2017'
data_dir=$server/'goAndGeneAnnotationMar2017/entailment_data/AicScore/go_bert_cls'
result_folder=$work_dir/'GCNOnto2vec/'$metric_option'.300' #$def_emb_dim.'clsVec'
model_load=$result_folder/'best_state_dict.pytorch'

word_mode='PretrainedGO' ## needed for GCN+something
w2v_emb='/local/datdb/Onto2Vec/GOVectorData/2017/onto2vec_embeddings.pickle'

conda activate tensorflow_gpuenv
cd $server/GOmultitask

pair='FlyWorm' # 'HumanFly'
outDir=$server/'geneOrtholog'/$pair'Score/qnliFormatData17'

finalDir=$outDir/$'GCNOnto2vec300'
mkdir $finalDir

for point in {0..3600..300} ;  do 
  echo ' '
  echo 'iter '$point

  savePickle=$outDir/'GeneDict2test.'$point'.pickle'

  saveDf=$outDir/'Ortholog2testDef.'$point'.txt'
  test_file=$saveDf
  write_score=$finalDir/'Ortholog2testDef.'$point'.score.txt'

  ## set epoch=0 for testing
  CUDA_VISIBLE_DEVICES=6 python3 $server/GOmultitask/GCN/encoder/do_model.py --main_dir $work_dir --qnli_dir $data_dir --batch_size_aa_go 128 --result_folder $result_folder --epoch 0 --use_cuda --metric_option $metric_option --nonlinear_gcnn $nonlinear_gcnn --def_emb_dim $def_emb_dim --model_load $model_load --test_file $test_file --write_score $write_score --w2v_emb $w2v_emb --word_mode $word_mode > $result_folder/test1.log

  paste $test_file $write_score > $finalDir/'score.'$point'.txt' ## append columns 

  # rm -f $test_file
  rm -f $write_score

done



#### *****


server='/local/datdb'

## use cosine similarity as objective function 
def_emb_dim='300'
metric_option='cosine'
nonlinear_gcnn='relu'

server='/local/datdb'
work_dir=$server/'goAndGeneAnnotationMar2017'
data_dir=$server/'goAndGeneAnnotationMar2017/entailment_data/AicScore/go_bert_cls'
result_folder=$work_dir/'GCNOnto2vec/'$metric_option'.300' #$def_emb_dim.'clsVec'
model_load=$result_folder/'best_state_dict.pytorch'

word_mode='PretrainedGO' ## needed for GCN+something
w2v_emb='/local/datdb/Onto2Vec/GOVectorData/2017/onto2vec_embeddings.pickle'

conda activate tensorflow_gpuenv
cd $server/GOmultitask

pair='Human'
outDir=$server/$pair'PPI3ontology/qnliFormatData17'

finalDir=$outDir/$'GCNOnto2vec300'
mkdir $finalDir


for point in {0..5300..50}  # 11700 5300..50
do 

  echo ' '
  echo 'iter '$point

  savePickle=$outDir/'GeneDict2test.'$point'.pickle'

  saveDf=$outDir/'PPI2testDef.'$point'.txt'
  test_file=$saveDf
  write_score=$finalDir/'PPI2testDef.'$point'.score.txt'

  ## set epoch=0 for testing
  CUDA_VISIBLE_DEVICES=6 python3 $server/GOmultitask/GCN/encoder/do_model.py --main_dir $work_dir --qnli_dir $data_dir --batch_size_aa_go 128 --result_folder $result_folder --epoch 0 --use_cuda --metric_option $metric_option --nonlinear_gcnn $nonlinear_gcnn --def_emb_dim $def_emb_dim --model_load $model_load --test_file $test_file --write_score $write_score --w2v_emb $w2v_emb --word_mode $word_mode > $result_folder/test1.log


  paste $test_file $write_score > $finalDir/'score.'$point'.txt' ## append columns 

  # rm -f $test_file
  rm -f $write_score

done




