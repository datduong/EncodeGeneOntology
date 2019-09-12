
module load python/3.7.2 
work_dir='/u/flashscratch/d/datduong/goAndGeneAnnotationDec2018'
data_dir='/u/flashscratch/d/datduong/goAndGeneAnnotationDec2018/entailment_data/AicScore'
## make pairs based on paper in 2018
cd /u/flashscratch/d/datduong/GOmultitask/BERT
python3 make_go_pair_cls.py $work_dir $data_dir

cd /u/flashscratch/d/datduong/GOmultitask/BERT
python3 make_go_pair_cls_cosine.py $work_dir $data_dir


## **** using our old data 2017
module load python/3.7.2 
work_dir='/u/flashscratch/d/datduong/goAndGeneAnnotationMar2017'
data_dir='/u/flashscratch/d/datduong/goAndGeneAnnotationMar2017/entailment_data/AicScore'
cd /u/flashscratch/d/datduong/GOmultitask/BERT
python3 make_go_pair_cls_cosine.py $work_dir $data_dir

## **** using our old data 2017, note::: how we make entailment data is different than cosine
module load python/3.7.2 
work_dir='/u/flashscratch/d/datduong/goAndGeneAnnotationMar2017'
data_dir='/u/flashscratch/d/datduong/goAndGeneAnnotationMar2017/entailment_data/AicScore'
cd /u/flashscratch/d/datduong/GOmultitask/BERT
python3 make_go_pair_cls.py $work_dir $data_dir


## **** using deepgo data 2016-2017 era
module load python/3.7.2 
work_dir='/u/flashscratch/d/datduong/deepgo/data'
data_dir='/u/flashscratch/d/datduong/deepgo/data/entailment_data/AicScore'
cd /u/flashscratch/d/datduong/GOmultitask/BERT
python3 make_go_pair_cls_cosine.py $work_dir $data_dir




## run entailment based on BERT. using QNLI as template input 
conda activate tensorflow_gpuenv
cd /local/datdb/pytorch-pretrained-BERT/examples/
data_dir='/local/datdb/goAndGeneAnnotationDec2018/entailment_data/AicScore/go_bert_cls/'
output_dir='/local/datdb/goAndGeneAnnotationDec2018/entailment_data/AicScore/go_bert_cls/model_cased_bioBERT/'
bert_model='/local/datdb/BERTPretrainedModel/pubmed_pmc_470k/'
CUDA_VISIBLE_DEVICES=4 /local/datdb/anaconda3/envs/tensorflow_gpuenv/bin/python -u run_classifier.py --data_dir $data_dir --output_dir $output_dir --bert_model $bert_model --bert_model_tokenizer bert-base-cased --task_name qnli --fp16 --num_train_epochs 4 --do_train --max_seq_length 384 --train_batch_size 16

## eval 
bert_model=$output_dir
CUDA_VISIBLE_DEVICES=7 /local/datdb/anaconda3/envs/tensorflow_gpuenv/bin/python -u run_classifier.py --data_dir $data_dir --output_dir $output_dir --bert_model $bert_model --bert_model_tokenizer bert-base-cased --task_name qnli --fp16 --num_train_epochs 4 --do_eval --max_seq_length 384 --train_batch_size 24

## now we lm fine tune, and then we do the QNLI task 
## make data for training (about 30 mins)
## use bert-base-cased because bioBERT uses this. 
cd /local/datdb/pytorch-pretrained-BERT/examples/lm_finetuning
python3 pregenerate_training_data.py --train_corpus /local/datdb/goAndGeneAnnotationDec2018/BERT_go_branch.txt --bert_model bert-base-cased --output_dir /local/datdb/goAndGeneAnnotationDec2018/BERT_base_cased_tune_go_branch/ --epochs_to_generate 5 --max_seq_len 384

## tune 
conda activate tensorflow_gpuenv
bert_model='/local/datdb/BERTPretrainedModel/pubmed_pmc_470k/'
cd /local/datdb/pytorch-pretrained-BERT/examples/lm_finetuning
CUDA_VISIBLE_DEVICES=1 /local/datdb/anaconda3/envs/tensorflow_gpuenv/bin/python -u finetune_on_pregenerated.py --pregenerated_data /local/datdb/goAndGeneAnnotationDec2018/BERT_base_cased_tune_go_branch/ --bert_model_tokenizer bert-base-cased --bert_model $bert_model --output_dir /local/datdb/goAndGeneAnnotationDec2018/BERT_base_cased_tune_go_branch/finetuned_lm_bioBERT/ --epochs 5 --train_batch_size 24 --fp16 

# QNLI by loading in the fine tune lm 
conda activate tensorflow_gpuenv
cd /local/datdb/pytorch-pretrained-BERT/examples/
data_dir='/local/datdb/goAndGeneAnnotationMar2017/entailment_data/AicScore/go_bert_cls/'
output_dir='/local/datdb/goAndGeneAnnotationMar2017/BERT_base_cased_tune_go_branch/fine_tune_lm_bioBERT/qnli_classifier/'
bert_model='/local/datdb/goAndGeneAnnotationMar2017/BERT_base_cased_tune_go_branch/fine_tune_lm_bioBERT/'
CUDA_VISIBLE_DEVICES=3 /local/datdb/anaconda3/envs/tensorflow_gpuenv/bin/python -u run_classifier.py --data_dir $data_dir --output_dir $output_dir --bert_model $bert_model --bert_model_tokenizer bert-base-cased --task_name qnli --fp16 --num_train_epochs 20 --do_train --max_seq_length 384 --train_batch_size 20

# eval
conda activate tensorflow_gpuenv
cd /local/datdb/pytorch-pretrained-BERT/examples/
data_dir='/local/datdb/goAndGeneAnnotationDec2018/entailment_data/AicScore/go_bert_cls/'
output_dir='/local/datdb/goAndGeneAnnotationDec2018/BERT_base_cased_tune_go_branch/finetuned_lm_bioBERT/model_cased_bioBERT/'
bert_model=$output_dir
CUDA_VISIBLE_DEVICES=7 /local/datdb/anaconda3/envs/tensorflow_gpuenv/bin/python -u run_classifier.py --data_dir $data_dir --output_dir $output_dir --bert_model $bert_model --bert_model_tokenizer bert-base-cased --task_name qnli --fp16 --num_train_epochs 5 --do_eval --max_seq_length 384 --train_batch_size 24



