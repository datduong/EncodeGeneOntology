


from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata, string, re, sys, os, pickle, pickle, gzip

from tqdm import tqdm

import numpy as np
import pandas as pd

from copy import deepcopy

import networkx
import obonet

sys.path.append('/u/flashscratch/d/datduong/GOmultitask')
from compare_set import compare_set

import GoAic


def ic2dict(ic): ## gaf file to dict {gene:[go]}
  df = pd.read_csv(ic,sep="\t", header=None)
  df.columns = ['GO','IC']
  # https://stackoverflow.com/questions/26684199/long-format-pandas-dataframe-to-dictionary
  return {g: float( d['IC'].values.tolist()[0] ) for g, d in df.groupby('GO')}


work_dir = '/u/flashscratch/d/datduong/goAndGeneAnnotationMar2017/'
os.chdir (work_dir)

go_name_array = pd.read_csv("go_name_in_obo.csv",header=None)
go_name_array = list ( go_name_array[0] )


# Read the taxrank ontology
graph = obonet.read_obo('go.obo') # https://github.com/dhimmel/obonet

AncestorGO = {}
for node in go_name_array:
  AncestorGO[node] = list ( networkx.descendants(graph, node) ) ## return parents


def submitJobs (onto_type) :

  # onto_type = 'CC'
  # species = 'Worm'

  for species in ['Human','Mouse','Fly','Worm','Yeast']:

    print ('\n\nspecies {}'.format(species))

    IcGO = ic2dict('ICdata/'+species+'Ic'+onto_type+'.txt')
    ## compare 2 GO terms, we want to see what the range is, and how IC can affect each species

    input_file = 'random_go_analysis_'+onto_type.lower()+'.tsv'
    fout = open ('RandomGOAnalysis/'+species+"_"+input_file,"w")
    fout.write ( "go1\tgo2\tscore\tic1\tic2\tlabel\ttype")
    df = pd.read_csv(input_file,sep='\t')


    for index,row in tqdm ( df.iterrows() ) :
      ## notice in using WormCC we see GO:0070724  GO:0005893  1.0000000000000002. Both terms share the same set of ancestors, but not appear in Yeast annotation.
      ## **** need to remove GO not found in species annotation
      # if ( row['go1'] not in IcGO ) or ( row['go2'] not in IcGO ):
      #   continue
      score = GoAic.Aic2GO ( row['go1'], row['go2'], IcGO, AncestorGO )
      ic1 = -1
      if row['go1'] in IcGO:
        ic1 = IcGO[row['go1']]

      ic2 = -1
      if row['go2'] in IcGO:  ## can have Inf ic
        ic2 = IcGO[row['go2']]

      fout.write( "\n" + row['go1'] + "\t" + row['go2'] + "\t" + str(score) + "\t" + str(ic1) + "\t" + str(ic2) + "\t" + row['label'] + "\t" + row['type'])

    fout.close()



if len(sys.argv)<1: ## run script
	print("Usage: \n")
	sys.exit(1)
else:
	submitJobs ( sys.argv[1] )


# GO:0030684  GO:0055029  0.4827797011194412  5.7655153 5.6670753 not_entailment  cc