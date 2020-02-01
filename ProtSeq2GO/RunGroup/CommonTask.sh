
#### scp to hoffman

for model in ; do 
  scp -r /local/datdb/deepgo/data/train/fold_1/DeepGOFlatSeqConcatGo$model $hoffman2:$scratch/deepgo/data/train/fold_1
done 

