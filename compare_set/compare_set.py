from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string, re, sys, os, pickle, gzip
from tqdm import tqdm
import numpy as np
import pandas as pd
from collections import namedtuple
from tempfile import TemporaryDirectory


def uniq(lst):
  last = object()
  for item in lst:
    if item == last:
      continue
    yield item
    last = item


def sort_and_deduplicate(l):
  return list(uniq(sorted(l, reverse=True)))


def hausdorff_distance (score): # @score is matrix
  ## compute the distance for both gene1->gene2 and also gene2->gene1
  ## !! WORKS ONLY IF METRIC IS SYMMETRIC TERM1 VS TERM2
  ## in our case, symmetric because cosine or max(A-->B , B-->A)
	rowMax = np.amax(score,1) ## max for each row ... same as max_b s(a,b) 
	rowMean = np.mean(rowMax) ## mean_a max_b s(a,b)
	colMax = np.amax(score,0) ## max for each col
	colMean = np.mean(colMax)
	return np.min([rowMean,colMean]) , np.mean([rowMean,colMean])  # hausdorff distance using both min/mean


class GenePair () :
  # takes 2 genes, create lookup extraction index
  def __init__(self,gene1,gene2,GeneGOdb1,GeneGOdb2):
    self.gene1 = gene1
    self.gene2 = gene2
    self.pair_wise(GeneGOdb1,GeneGOdb2)

  def pair_wise(self,GeneGOdb1,GeneGOdb2):
    self.shape = (len(GeneGOdb1[self.gene1]), len(GeneGOdb2[self.gene2]))
    self.GOpair = []
    for i in GeneGOdb1[self.gene1]:
      for j in GeneGOdb2[self.gene2]:
        # pair = sorted ( [i,j] ) ## check reverse case. for cosine-distance, we do not need to compare A->B and then B->A
        # pair = pair[0] + "," + pair[1]
        # hausdorff distance needs i-->j and j-->i
        # case gene1=[a,b] gene2=[b] and gene3=[b,a] gene4=[a], will have same GO pair [ab] [ba]. but we will not reduce down to only [ab] ... too hard to backtrack
        self.GOpair.append ( i + "," + j )


class GenePairDict ():
  def __init__(self,genePairList): ## gene pairs to compare
    # we will read the genePairList outside of this class
    # self.genePairList = pd.read_csv(genePairList,sep=" ",dtype=str,header=None)
    self.genePairList = genePairList
    self.genePairList.columns = ['gene1','gene2','label']

  def make_pair (self,GeneGOdb1,GeneGOdb2):

    geneSet1 = list (self.genePairList['gene1'])
    geneSet2 = list (self.genePairList['gene2'])
    label = list (self.genePairList['label'])

    self.genePair = {}
    self.LargeGOpair = {}

    for index in tqdm ( range (self.genePairList.shape[0]) ) :

      if (geneSet1[index] not in GeneGOdb1) or (geneSet2[index] not in GeneGOdb2):
        print ('skip {} , {}'.format(geneSet1[index], geneSet2[index]))
        continue

      thisPair = GenePair (geneSet1[index],geneSet2[index],GeneGOdb1,GeneGOdb2) ## create gene pair object
      self.genePair[ geneSet1[index]+","+geneSet2[index] ] = [ thisPair, label[index] ] ## add to list that we have seen

      for p in thisPair.GOpair: ## GO pairs for these 2 genes
        if p not in self.LargeGOpair:
          self.LargeGOpair[p] = 0 ## we will put score here later. for now, put 0

  def get_index (self):
    for thisPair in self.genePair: # has object @pair and @label
      self.genePair[thisPair][0].get_index( self.LargeGOpair ) ## now, each GOPair will have a set of index to extract

  def write_go_pairs(self,GOdef,nameOut): # make same input as how we train the model
    # index question  sentence  go1 go2 label
    fout = open (nameOut,'w')
    fout.write('index\tquestion\tsentence\tgo1\tgo2\tlabel')
    counter = 0
    for key,val in self.LargeGOpair.items() :
      p = key.split(",") ## get GO terms
      d1 = " ".join(w for w in GOdef[p[0]])
      d2 = " ".join(w for w in GOdef[p[1]])
      # p = [re.sub('GO:',"",go) for go in p] ## keep this same format as original entailment input
      fout.write('\n'+str(counter)+'\t'+d1+'\t'+d2+'\t'+p[0]+'\t'+p[1]+'\tnot_entailment')
      counter = counter + 1

    fout.close()

  def read_score (self,ScoreFile): ## @ScoreFile is a csv : go1,go2,score, we will reconstruct the GO terms
    ScoreFile = pd.read_csv(ScoreFile,sep="\t")
    goSet1 = list (ScoreFile['go1'])
    goSet2 = list (ScoreFile['go2'])
    score = list (ScoreFile['score'])

    # reuse @self.LargeGOpair
    for i in range(len(goSet1)) :
      self.LargeGOpair [ goSet1[i]+","+goSet2[i] ] = score[i] # get score for each pair of GO term

  def score_gene (self,nameOut):
    fout = open(nameOut,"w")
    # fout.write('gene1,gene2,trueLabel,score1,score2')
    for g in self.genePair :
      this = self.genePair[g][0] ## @g is array [pair-object , is_same label] so we get first entry
      score_array = [ self.LargeGOpair[s] for s in this.GOpair ] ## get GO1-GO2 for these 2 genes
      ## convert the array into a matrix
      ## notice, fill by ROW
      score_array = np.reshape (score_array,this.shape)
      score_array = score_array.astype(float)
      ## call hausdorff score
      final_score = hausdorff_distance (score_array) ## return tuple
      self.genePair[g] = self.genePair[g] + list(final_score) ## append ... cost time, so we should just keep it ?
      fout.write( this.gene1 + "\t" + this.gene2 + "\t" + self.genePair[g][1] + "\t" + "\t".join( str(s) for s in final_score) + "\n" ) # "\n"+ 

    # end
    fout.close() 


def gaf2dict(gaf): ## gaf file to dict {gene:[go]}
  df = pd.read_csv(gaf,sep="\t",dtype=str)
  # https://stackoverflow.com/questions/26684199/long-format-pandas-dataframe-to-dictionary
  # sort GO terms so backtracking is easier
  return {g: sorted(d['go'].values.tolist()) for g, d in df.groupby('uniprot')} 


def def2dict(name="go_def_in_obo.tsv"):
  GOdef = pd.read_csv(name,dtype=str,sep="\t")
  return {g: d['def'].values.tolist() for g, d in GOdef.groupby('name')}

