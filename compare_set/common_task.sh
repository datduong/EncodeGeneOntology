


for where in HumanFlyScore HumanMouseScore FlyWormScore MouseFlyScore ; do #HumanFlyScore HumanMouseScore FlyWormScore
  for what in 'cosine768Linear768Layer11+12nnParamOpW2' ; do 
    scp -r /local/datdb/geneOrtholog/$where/qnliFormatData17/$what $hoffman2:$scratch/geneOrtholog/$where/qnliFormatData17
  done
done 




for where in HumanPPI3ontology YeastPPI3ontology ; do #HumanFlyScore HumanMouseScore FlyWormScore
  for what in cosine768Linear768Layer11+12nnParamOpW2 ; do 
    echo ' '
    echo ' '$where' '$what
    echo ' '
    scp -r /local/datdb/$where/qnliFormatData17/$what $hoffman2:$scratch/$where/qnliFormatData17
  done
done 





for where in HumanFlyScore HumanMouseScore MouseFlyScore FlyWormScore; do 
  for what in 'cosine.bilstm768' 'cosine.AveWordClsSep768.Linear768.Layer12' ; do 
    rm -rf /u/flashscratch/d/datduong/geneOrtholog/$where/qnliFormatData17/$what 
  done
done 


