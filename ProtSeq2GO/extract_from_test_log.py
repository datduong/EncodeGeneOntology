import re,os,sys,pickle

## seem stupid, but we have too many test output, too stupid to manual look up. 

# ['GCNOnto2vecAppendLast','NotUseGo2','BilstmGoRun2','GCNGoRun2','BertGoRun3','Onto2vec']
# ['BertAveWordClsSep','BertAveWordOnly','BertAsService','NotUseGo2','BilstmGoRun2','GCNGoRun2','BertGoRun3','Onto2vec']


for folder_to_look in ['FlatSeqProtHwayGCN768Lr0.001b32'] :
  #
  file_list = ['/local/datdb/deepgo/dataExpandGoSet/train/fold_1/FlatSeqProtHwayGCN768Lr0.001b64.bp/test_frequency.log', '/local/datdb/deepgo/dataExpandGoSet/train/fold_1/'+folder_to_look+'.mf/test_frequency.log','/local/datdb/deepgo/dataExpandGoSet/train/fold_1/FlatSeqProtHwayGCN768Lr0.001b64.cc/test_frequency.log']
  ## for each type 
  fout = open("/local/datdb/deepgo/dataExpandGoSet/train/fold_1/"+folder_to_look+".txt","w")
  for bucket in ['^round cutoff 0.5', '^less than quant25', '^less than quant75', '^less than betweenQ25Q75'] :
    fout.write("\n\n"+bucket+"\n\n")
    for f in file_list: 
      rec_at_k = [] 
      prec_at_k = []
      fin = open(f,"r")
      start_track = False ## set to not track any lines
      for line in fin: 
        if bool( re.match(bucket,line) ) : ## print out standard rounding threshold 0.5 # ^round cutoff 0.5 ^less than quant75 count betweenQ25Q75
          start_track = True
        if start_track: 
          if bool( re.match("^hamming loss",line) ) : ## track until next one 
            start_track = False
            fout.write("\n")
            break ## do not track any more
          else: 
            if bool( re.match("^rec_at_",line) ) : 
              print (line)
              rec_at_k.append(line.strip().split()[-1]) ## get last element
            if bool( re.match("^prec_at_",line) ) : 
              print (line)
              prec_at_k.append(line.strip().split()[-1]) ## get last element
      ##
      fout.write("\t".join(str(j) for j in rec_at_k)+"\n")
      fout.write("\t".join(str(j) for j in prec_at_k))
  ## close 
  fin.close() 



# ##
# import re,os,sys,pickle
# for folder_to_look in ['BertAveWordClsSep','BertAveWordOnly','BertAsService','NotUseGo2','BilstmGoRun2','GCNGoRun2','BertGoRun3','Onto2vec']:
#   #
#   file_list = ['/local/datdb/deepgo/data/train/fold_1/FlatSeqProtHway'+folder_to_look+'.bp/test_frequency.log', '/local/datdb/deepgo/data/train/fold_1/FlatSeqProtHway'+folder_to_look+'.mf/test_frequency.log','/local/datdb/deepgo/data/train/fold_1/FlatSeqProtHway'+folder_to_look+'.cc/test_frequency.log']
#   fout = open("/local/datdb/deepgo/data/train/fold_1/"+folder_to_look+".txt","w")
#   for f in file_list: 
#     rec_at_k = [] 
#     fin = open(f,"r")
#     start_track = False ## set to not track any lines
#     for line in fin: 
#       if bool( re.match("^less than quant25",line) ) : ## print out standard rounding threshold 0.5 # ^round cutoff 0.5 ^less than quant75 count
#         start_track = True
#       if start_track: 
#         if bool( re.match("^micro average prec",line) ) : 
#           print (line)
#           rec_at_k.append(line.strip().split()[-1]) ## get last element
#           start_track = False
#           fout.write("\n")
#           break ## do not track any more
#     ##
#     fout.write("\t".join(str(j) for j in rec_at_k)+"\n")
#     fin.close() 

