
server='/local/datdb'

## use cosine similarity as objective function 
def_emb_dim='300'
metric_option='cosine'

work_dir=$server/'goAndGeneAnnotationDec2018'
bert_model=$work_dir/'BERT_base_cased_tune_go_branch/fine_tune_lm_bioBERT' # use the full mask + nextSentence to innit
data_dir=$server/'goAndGeneAnnotationDec2018/entailment_data/AicScore/go_bert_cls'
pregenerated_data=$server/'goAndGeneAnnotationDec2018/BERT_base_cased_tune_go_branch' # use the data of full mask + nextSentence to innit
bert_output_dir=$pregenerated_data/'fine_tune_lm_bioBERT'
mkdir $bert_output_dir

result_folder=$bert_output_dir/$metric_option'.768.reduce300ClsVec' #$def_emb_dim.'clsVec'
mkdir $result_folder

model_load=$result_folder/'best_state_dict.pytorch'

conda activate tensorflow_gpuenv
cd $server/GOmultitask


# 54178
gapSize=300

genePairList=$server/'geneOrtholog/HumanMouseScore/HumanMouseOrtholog2test.txt'
outDir=$server/'geneOrtholog/HumanMouseScore/qnliFormat'
mkdir $outDir
finalDir=$outDir/$metric_option'.768.reduce300ClsVec'
mkdir $finalDir

gaf1='goa_human_not_IEA.tsv'
gaf2='mgi_not_IEA.tsv'

for point in {0..22500..300} # 55000
do 

echo ' '
echo 'iter '$point

savePickle=$outDir/'GeneDict2test.'$point'.pickle'

saveDf=$outDir/'Ortholog2testDef.'$point'.txt'
test_file=$saveDf
write_score=$finalDir/'Ortholog2testDef.'$point'.score.txt'


## set epoch=0 for testing
CUDA_VISIBLE_DEVICES=7 python3 $server/GOmultitask/BERT/encoder/do_joint_model.py --main_dir $work_dir --qnli_dir $data_dir --batch_size_label 64 --batch_size_bert 8 --bert_model $bert_model --pregenerated_data $pregenerated_data --bert_output_dir $bert_output_dir --result_folder $result_folder --epoch 0 --num_train_epochs_entailment 0 --use_cuda --metric_option $metric_option --def_emb_dim $def_emb_dim --reduce_cls_vec --model_load $model_load --write_score $write_score --test_file $test_file > $result_folder/test1.log

paste $test_file $write_score > $finalDir/'score.'$point'.txt' ## append columns 

# rm -f $test_file
rm -f $write_score

done


