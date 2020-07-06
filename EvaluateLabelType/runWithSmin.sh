
/local/datdb/deepgo/data/train/deepgo.'+onto+'.csv'

load_path = method+'/'+onto+'b32lr0.001
    prediction_dict = pickle.load(open(load_path+"/prediction_testset.pickle","rb"))



prediction_testset.pickle


####

### ! split by quantile counts, then get accuracy

server='/u/scratch/d/datduong' #/u/scratch/d/datduong/ /local/datdb
codepath=$server/'EncodeGeneOntology/EvaluateLabelType'
cd $codepath

for onto in 'mf' 'cc' 'bp' ; do # 'bp'

  for model in DeepGOFlatSeqConcatGoOnto2vec DeepGOFlatSeqConcatGoGCN DeepGOFlatSeqConcatGoBiLSTM DeepGOFlatSeqConcatGoBertGOName DeepGOFlatSeqConcatGoBertCLS12 ; do # 'DeepGOFlatSeqConcatGoBertAve12' ; do # 'BaseExtraLayer' Epo1000bz6

    label_path=$server/'deepgo/data/train/global-count-'$onto'.tsv' # global-count-mf.tsv

    model_path=$server/'deepgo/data/train/fold_1/'$model'/NotNormalize/'$onto'b32lr0.0005RMSprop'

    output=$model_path
    load_path=$output/prediction_testset.pickle

    python3 $codepath/EvalLabelQuantile.py $label_path $onto $load_path > $output/$onto.printout.txt

  done
done
cd $output

