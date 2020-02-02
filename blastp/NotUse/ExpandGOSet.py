


import pickle, os, sys, gzip, re
import pandas as pd
import numpy as np
from tqdm import tqdm

import gensim
from gensim.models import KeyedVectors

## take best set of GO, expand it by finding closet neighbors
## best to use gensim because it can easily find closest words/vectors. and also we have code to plot.

class GOVector ():

  def __init__(self, wordvec_file, label_subset_file, wordvec_file_small=None,cut_point=0.5, topk=10):

    # @wordvec_file is GO vector in gensim text format
    # @label_subset_file is list of GO we're testing on

    self.GetLabel (label_subset_file) ## create the @self.label_to_test
    self.topk = topk

    print ('keep only go terms needed ... unless we test against whole GO database')

    if wordvec_file_small is None:
      print ('using all GO terms in database')
      self.wordvec = KeyedVectors.load_word2vec_format(wordvec_file, binary=False)
    else:
      print ('remove GO terms not needed')
      self.FilterGO (wordvec_file, wordvec_file_small)
      self.wordvec = KeyedVectors.load_word2vec_format(wordvec_file_small, binary=False)

    self.cut_point = cut_point ## what if a GO term is super different from every other GO terms ???? is that possible ??

  def FilterGO (self, wordvec_file, wordvec_file_small):

    # https://stackoverflow.com/questions/48941648/how-to-remove-a-word-completely-from-a-word2vec-model-in-gensim ... but doesn't work

    # ### !!! keep only GO terms to be tested, some GO terms are not needed if we do some filtering.

    ## filter the input text file.
    ## stupid thing is that @KeyedVectors.load_word2vec_format reads in a .txt or a binary. so we have to do a read/write for @label_subset_file
    fin = open (wordvec_file,"r")
    fout = open(wordvec_file_small,"w")

    for i, line in enumerate(fin):

      line = line.strip().split()

      if i == 0:
        fout.write( str(len(self.label_to_test)) + " " + line[1] + "\n" )

      if line[0] in self.label_to_test:
        fout.write(" ".join(line)+"\n")

    fout.close()
    fin.close()

  def GetLabel (self,label_subset_file):
    df = pd.read_csv(label_subset_file, header=None)
    label_to_test = sorted ( list (df[0]) )
    self.label_to_test = {val:number for number, val in enumerate(label_to_test)}
    print ('total label to test {}\n\n'.format(len(self.label_to_test)))

  def GetNearestGO(self,this_go,this_score):
    # @this_go is "GO:xyz" @this_score is some number like 4.5
    topk_closest = self.wordvec.most_similar(this_go,topn=self.topk) # https://kavita-ganesan.com/gensim-word2vec-tutorial-starter-code/
    word_list_score = {}
    for word,score in topk_closest:
      if score > self.cut_point:
        word_list_score[word] = score * this_score ## scale down
    return word_list_score

  def GetNearestGOIter (self,go_array,score_array):

    ## iter over an array of GO and score, for each GO, call @GetNearestGO

    ## @score_array is the blast score
    min_score = 1 ## to avoid over powering blast prediction
    for go,score in zip(go_array,score_array):
      if score < min_score:
        min_score = score

    go_score_list = {}
    for go,score in zip(go_array,score_array):

      if go not in go_score_list:
        go_score_list[go] = score ## add self. we must do this, so that we have ALL the GO terms

      ## get top closest wrt @go.
      expand_topk = self.GetNearestGO(go, min_score) ## get more go terms

      for k,val in expand_topk.items():

        if k in go_array :
          continue ## can find blast GO in expanded set. Here, skip.

        if k not in go_score_list:
          go_score_list[k] = val
        else:
          if val < go_score_list[k]:
            go_score_list[k] = val ## update to lower score

    return go_score_list

  def ExpandBlast (self,blast_score_dict):

    ## more efficient to use @blast_score_dict as dictionary than as a matrix

    blast_score_dict_new = {} ## avoid pass by reference on @blast_score_dict
    for prot_name,predicted_go in tqdm ( blast_score_dict.items() ): ## can be slow.

      ## sample i, get the blast GO, and its score. then expand the GO set

      go_array_blast = []
      go_score_blast = []
      for go_name,go_score in predicted_go.items(): ## iter over score, we know blast give "yes" if score is not 0
        go_score_blast.append(go_score)
        go_array_blast.append(go_name)

      blast_score_dict_new[prot_name] = self.GetNearestGOIter(go_array_blast, go_score_blast)

    return blast_score_dict_new

  def dict2matrix (self,blast_score_dict,prot_name_array):

    ## !!!! @prot_name_array is array of prot/prot names, but they need to match the exact ordering in how BERT/GCN/BiLSTM were done.

    final_score = np.zeros((len(prot_name_array),len(self.label_to_test)))

    for this_prot in blast_score_dict: ## @blast_score_dict is {prot:{go,score}}

      go_index_blast = []
      go_score_blast = []

      for go_name,go_score in blast_score_dict[this_prot].items():
        go_score_blast.append(go_score)
        go_index_blast.append( self.label_to_test[go_name] ) ## which index to make non-zero

      final_score[ prot_name_array.index(this_prot) , go_index_blast ] = go_score_blast

    return final_score

  def truelabel2matrix (self,blast_score_dict,prot_name_array):
    ## convert true label dict into matrix
    final_score = np.zeros((len(prot_name_array),len(self.label_to_test)))

    for this_prot in blast_score_dict: ## @blast_score_dict is {prot:{go,score}}

      go_index_blast = []

      for go_name in blast_score_dict[this_prot]:
        go_index_blast.append( self.label_to_test[go_name] ) ## which index to make non-zero

      final_score[ prot_name_array.index(this_prot) , go_index_blast ] = 1

    return final_score






