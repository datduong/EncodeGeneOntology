import re,os,sys,pickle

## seem stupid, but we have too many test output, too stupid to manual look up. 

# file_list = ['/local/auppunda/auppunda/deepgo/data/train/fold_1/FlatSeqProtHwayBertGo.bp/gg.001_24/test_frequency.log', '/local/auppunda/auppunda/deepgo/data/train/fold_1/FlatSeqProtHwayBertGo.mf/gg.001_24/test_frequency.log','/local/auppunda/auppunda/deepgo/data/train/fold_1/FlatSeqProtHwayBertGo.cc/gg.001_24/test_frequency.log']
# fout = open("/local/datdb/deepgo/data/train/fold_1/BertAveToken+CLS+SEP.txt","w")

# file_list = ['/local/datdb/deepgo/data/train/fold_1/FlatSeqProtHwayGCNGoRun2.bp/test_frequency.log', '//local/datdb/deepgo/data/train/fold_1/FlatSeqProtHwayGCNGoRun2.mf/test_frequency.log','/local/datdb/deepgo/data/train/fold_1/FlatSeqProtHwayGCNGoRun2.cc/test_frequency.log']
# fout = open("/local/datdb/deepgo/data/train/fold_1/GCN.txt","w")


file_list = ['/local/auppunda/auppunda/deepgo/data/train/fold_1/FlatSeqProtHwayBertGo.bp/fff/test_frequency.log', '/local/auppunda/auppunda/deepgo/data/train/fold_1/FlatSeqProtHwayBertGo.mf/fff/test_frequency.log','/local/auppunda/auppunda/deepgo/data/train/fold_1/FlatSeqProtHwayBertGo.cc/fff/test_frequency.log']
fout = open("/local/datdb/deepgo/data/train/fold_1/BertAveTokenWordOnly.txt","w")


for f in file_list: 
  fin = open(f,"r")
  start_track = False ## set to not track any lines
  for line in fin: 
    if bool( re.match("^round cutoff 0.5",line) ) : ## print out standard rounding threshold 0.5 
      start_track = True
    if start_track: 
      if bool( re.match("^round cutoff 0.6",line) ) : ## track until next one
        start_track = False
        fout.write("\n")
        break ## do not track any more
      else: 
        fout.write(line)
    ##
  ## close 
  fin.close() 


