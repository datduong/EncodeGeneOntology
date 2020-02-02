import re,os,sys,pickle

fout = open("/local/datdb/deepgo/data/train/fold_1/DeepGOFlatSeqProtHwayGoNotUsePPI.test.txt","w")
name_list = 'GCN Onto2vec BertGOName BiLSTM Bertd11d12 BertAve12 UniformGOVector BertCLS12 BertAsService'.split()
for name in name_list: # 'BiLSTM', 'Bertd11d12', 'GCN', 'BertAve12' Base
  ## COMMENT write out result so we can paste into excel
  for onto in ['bp','mf','cc']: 
    try:
      fin = open("/local/datdb/deepgo/data/train/fold_1/DeepGOFlatSeqProtHwayGoNotUsePPI"+name+"/NotNormalize/"+onto+"b32lr0.0005RMSprop/test_frequency.log","r")
    except:
      continue
    print ('\nmethod {} ontology {}\n'.format(name,onto))
    fout.write('\nmethod {} ontology {}\n'.format(name,onto))
    do_print = False
    for line in fin :
      if ('[MACRO]' in line) or ('fmax' in line): 
        do_print = True ## turn on printing
      if ('rec_at_' in line) or ('macro average' in line): 
        do_print = False
      if do_print: 
        fout.write (line)
    ##!!
    fin.close()
#

fout.close()


