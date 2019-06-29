
import pickle,gzip,os,sys,re
import pandas as pd
import numpy as np

from tqdm import tqdm 

from pytorch_pretrained_bert.tokenization import BertTokenizer, load_vocab, whitespace_tokenize

main_dir = '/u/scratch/d/datduong/goAndGeneAnnotationDec2018/'
os.chdir(main_dir)

## we have already did this before. just need to load the words back
## but let's redo because it's important how we tokenize
## use white space tokenzier? 
## bert style when possible?

## format is one word per line

def whitespace_tokenizer (vocab,text_a): 
  # we follow same tokenization approach in original paper
  # @Vocab is some object that convert words to index exactly in the order of the pretrained word vectors. 
  # use Vocab = load_vocab('/local/datdb/MIMIC3database/format10Jan2019/vocab+icd_index_map.txt')
  # @text_a can be split by space (in case of preprocessed icd9 notes)
  tokens_a = whitespace_tokenize(text_a) 
  input_1_ids = []
  for token in tokens_a:
    if token in vocab:
      input_1_ids.append(vocab[token])
    # else: 
    #   input_1_ids.append(1) # unknown 
  return input_1_ids, len(input_1_ids)


## create the intersection list

## read pubmed
pubmed_vocab = []
fin = open("/u/scratch/d/datduong/w2vModel1Gram9Jan2019/vocab.txt","r")
counter = 0
for line in tqdm(fin): 
  if counter == 0: 
    counter = 1 # skip header
    continue
  pubmed_vocab.append ( line.split()[0] )

fin.close() 
pubmed_vocab = set (pubmed_vocab)

## read in def, do white space, intersect with pubmed 
GOdb_vocab = []
GOdb = pd.read_csv ( "go_def_in_obo.tsv", sep="\t" )
for defin in list(GOdb['def']): 
  token = whitespace_tokenize ( defin )
  token = list ( set (token) ) 
  GOdb_vocab = GOdb_vocab + token 

GOdb_vocab = set (GOdb_vocab)

GOdb_vocab = GOdb_vocab.intersection(pubmed_vocab)
GOdb_vocab = list(GOdb_vocab)
GOdb_vocab.sort()
GOdb_vocab = ['[PAD]','[UNK]'] + GOdb_vocab ## ADD PADDING

fout = open('word_pubmed_intersect_GOdb.txt','w')
fout.write("\n".join(s for s in GOdb_vocab))
fout.close() 


## create init embed. 

GOdb_vocab = [] 
fin = open ("word_pubmed_intersect_GOdb.txt",'r')
for line in fin: 
  GOdb_vocab.append( line.strip() )


fin.close() 


print ('\n\nextract emb as np')
word_dim = 300
pretrain_emb = np.zeros ( (len(GOdb_vocab), word_dim ) ) 

fin = open ("/u/scratch/d/datduong/w2vModel1Gram9Jan2019/w2vModel1Gram9Jan2019.txt",'r')
counter = 0 
for line in tqdm(fin): 
  if counter == 0 : 
    counter = 1 # skip line 1
    continue
  line = line.strip().split() 
  word = line[0]
  if word in GOdb_vocab: # of word not found, then we set 0 to that word
    pretrain_emb [ GOdb_vocab.index(word) ] = np.array ( line[1:len(line)] )


fin.close() 
print ('sample emb')
print (pretrain_emb[0:4])
pickle.dump(pretrain_emb,open("word_pubmed_intersect_GOdb_w2v_pretrained.pickle","wb"))


exit() 



where = "w2vDim300/EntDataJan19w300Base/word_indexing_for_entailment.pickle"
vocab = pickle.load(open(where,"rb")) ## {index:word}

## sort by alphabet ?? ## we already have indexing here
vocab_array = np.zeros(len(vocab)).tolist()

for index,word in enumerate(vocab):
  vocab_array[index] = word ## put to array by index

#
fout = open('word_pubmed_intersect_GOdb.txt','w')
fout.write("\n".join(s for s in vocab_array))
fout.close() 

