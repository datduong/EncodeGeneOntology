
from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string, re, sys, os, pickle, gzip
from tqdm import tqdm
import numpy as np
import pandas as pd
from collections import namedtuple
from tempfile import TemporaryDirectory

import compare_set

main_dir = '/u/scratch/d/datduong/goAndGeneAnnotationDec2018'
os.chdir(main_dir)


def submitJobs (pickleDf,scoreFile,nameOut,start):

  geneDict = pickle.load(open(pickleDf,'rb')) 
  geneDict.read_score (scoreFile)
  geneDict.score_gene (nameOut)
  
if len(sys.argv)<1: ## run script
	print("Usage: \n")
	sys.exit(1)
else:
	submitJobs ( sys.argv[1] , sys.argv[2] , sys.argv[3] , int(sys.argv[4]) )


