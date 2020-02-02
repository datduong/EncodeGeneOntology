

#### run on 'have seen vs unseen' when we expand the dataset
codepath='/local/datdb/GOmultitask/EvaluateLabelType'
output='/local/datdb/deepgo/data/train/fold_1/'
folder='/local/datdb/deepgo/data/train/fold_1/'
# GCN Onto2vec BertGOName BiLSTM Bertd11d12 BertAve12 ELMO ELMONotNormalize
# UniformGOVector BertCLS12 BertAsService 
for model in UniformGOVector ; do 
  run_model='DeepGOFlatSeqConcatGo'
  cd $codepath
  model_path=$folder$run_model$model
  # test-"+onto+"-same-origin.pickle"
  python3 $codepath/EvalLabelUnseen.py $output $model_path/NotNormalizeSgd0.00010.001 'b32lr0.0001RMSprop' > $output/$run_model$model.NotNormalize16Jan20Run4.txt
done
cd $output


#### clean up output, move stuffs into folders

output='/local/datdb/deepgo/data/train/fold_1/'
run_model='DeepGOFlatSeqConcatGo'
for model in GCN Onto2vec BertGOName BiLSTM Bertd11d12 BertAve12 UniformGOVector BertCLS12 BertAsService  ; do
  cd $output$run_model$model
  mkdir Normalize
  mv *b32lr0.0005RMSprop Normalize
done



#### run on split by quantile counts
codepath='/local/datdb/GOmultitask/EvaluateLabelType'
output='/local/datdb/deepgo/data/train/fold_1/'
run_model='DeepGOFlatSeqProtHwayGoNotUsePPI' #
folder='/local/datdb/deepgo/data/train/fold_1/'$run_model
for model in 'Bertd11d12'; do # 'BaseExtraLayer'
  cd $codepath
  model_path=$folder$model
  python3 $codepath/EvalLabelQuantile.py $model_path > $output/$run_model$model.txt
done


#### run on split by quantile counts ... BASEline
codepath='/local/datdb/GOmultitask/EvaluateLabelType'
output='/local/datdb/deepgo/dataExpandGoSet16Jan2020/train/fold_1/'
folder='/local/datdb/deepgo/dataExpandGoSet16Jan2020/train/fold_1/'
# GCN Onto2vec BertGOName BiLSTM Bertd11d12 BertAve12 
# UniformGOVector BertCLS12 BertAsService 
for model in Base ; do 
  run_model='DeepGOFlatSeqOnly'
  cd $codepath
  model_path=$folder$run_model$model
  # test-"+onto+"-same-origin.pickle"
  python3 $codepath/EvalLabelUnseen.py $output $model_path/ExtraLayer 'b32lr0.001RMSprop' > $output/$run_model$model.Base.txt
done
cd $output


