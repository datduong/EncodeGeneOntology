import re,os,sys,pickle

rec_at_k = '' ##!! start fresh
prec_at_k= ''

for name in ['Base' ]: # 'BiLSTM', 'Bertd11d12', 'GCN', 'BertAve12' 
  ## COMMENT write out result so we can paste into excel
  fin = open("/local/datdb/deepgo/dataExpandGoSet16Jan2020/train/fold_1/DeepGOFlatSeqProtHwayNotUseGo"+name+".txt","r")
  fout = open("/local/datdb/deepgo/dataExpandGoSet16Jan2020/train/fold_1/DeepGOFlatSeqProtHwayNotUseGo"+name+".out.txt","w")
  for line in fin : 
    # if 'type ' in line : 
    #   fout.write ('\n'+line+"\n") ##!! new line because new ontology 
    if line.strip() in ['whole','original','added'] : 
      # fout.write ("\n"+line+"\n")
      rec_at_k = '' ##!! start fresh for each bucket
      prec_at_k= ''
    if bool( re.match("^hamming loss",line) ) : ## track until next one
      fout.write(rec_at_k.strip()+'\n')
      fout.write(prec_at_k.strip()+'\n')
      # fout.write("\n") ## do not track any more
    else:
      if bool( re.match("^rec_at_",line) ) :
        rec_at_k = rec_at_k + " " + line.strip().split()[-1] ## get last element
      if bool( re.match("^prec_at_",line) ) :
        prec_at_k = prec_at_k + " " + line.strip().split()[-1] ## get last element
  ##!!
  fin.close() 
  fout.close()
