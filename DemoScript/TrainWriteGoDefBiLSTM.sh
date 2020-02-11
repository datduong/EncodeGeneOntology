

#### train and write go def into vectors

##!! you will need to change folder paths

def_emb_dim='768' ## final dim of GO vectors
metric_option='cosine' ## we only support cosine similarity right now, but I guess there are other metric types.
server='/local/datdb'

#### you can download our data from google drive.
work_dir=$server/'goAndGeneAnnotationMar2017' ## main working dir
data_dir=$server/'goAndGeneAnnotationMar2017/entailment_data/AicScore/go_bert_cls' ## train/dev/test samples

#### you can download our w2v from google drive.
w2v_emb=$work_dir'/word_pubmed_intersect_GOdb_w2v_pretrained.pickle' ## word2vec input
vocab_list='word_pubmed_intersect_GOdb.txt' # all vocab in GO database

result_folder=$work_dir/ExampleOutputDim$def_emb_dim
mkdir $result_folder

conda activate tensorflow_gpuenv ## may not exactly need tensorflow

cd $server/EncodeGeneOntology

#### train bilstm model
##!! you can ignore this step, if you only need to use pre-trained model
CUDA_VISIBLE_DEVICES=0 python3 $server/EncodeGeneOntology/biLSTM/encoder/do_model.py --fix_word_emb --vocab_list $vocab_list --w2v_emb $w2v_emb --lr 0.0001 --bilstm_dim 1024 --main_dir $work_dir --qnli_dir $data_dir --batch_size_aa_go 32 --result_folder $result_folder --epoch 100 --use_cuda --metric_option $metric_option --def_emb_dim $def_emb_dim --reduce_cls_vec > $result_folder/train.log

#### write vec

label_desc_dir=$work_dir/'go_def_in_obo.tsv'
model_load=$result_folder/'best_state_dict.pytorch'

CUDA_VISIBLE_DEVICES=0 python3 $server/EncodeGeneOntology/biLSTM/encoder/do_model.py --fix_word_emb --vocab_list $vocab_list --w2v_emb $w2v_emb --lr 0.0001 --bilstm_dim 1024 --main_dir $work_dir --qnli_dir $data_dir --batch_size_aa_go 32 --result_folder $result_folder --epoch 0 --num_train_epochs_entailment 0 --use_cuda --metric_option $metric_option --def_emb_dim $def_emb_dim --reduce_cls_vec --model_load $model_load --write_vector --label_desc_dir $label_desc_dir > $result_folder/write.log



