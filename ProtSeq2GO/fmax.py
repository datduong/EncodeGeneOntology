


import sys, re, random, string, json, operator, time # pyLDAvis
import numpy as np


## using f-max which is a variation of the f-measure (not f1-score)

def pr_rc_it (truth, prob, t):

  yhat = np.where ( prob > t ) [0] ## prediction probabilities

  if (len(yhat) == 0) or (len(truth) == 0) :
    return 0 , 0

  truth = np.where (truth > 0) [0] ## indexing of true labels
  if len(truth) == 0: ## sample with no labels
    return 0 , 0

  num = len ( set (yhat).intersection ( set(truth) ) ) * 1.0
  pr_i = num / len(yhat)
  ## recall
  rc_i = num / len(truth)
  return pr_i, rc_i


def pr_rc_t (pr_t, rc_t): # $pr_t is array over prot.
  ## for each protein
  mt = np.where ( pr_t > 0 ) [0]
  if len(mt) == 0 : # @mt is number of proteins on which at least one prediction was made above threshold t.
    pr_t = 0
  else:
    pr_t = np.mean ( pr_t [mt] )
  rc_t = np.mean ( rc_t ) ## over all protein n

  f = 2 * pr_t * rc_t / ( pr_t + rc_t )
  return f


def f_max ( true_set, prob, threshold=np.arange(0.005,1,.01) ) :
  # @true_set, @prob are np.2d-array
  # @threshold is np.vector
  f_value = np.zeros( len(threshold) )
  counter = -1

  for t in threshold :
    counter = counter + 1
    pr_t = []
    rc_t = []

    for prot in range(true_set.shape[0]):
      pr_i , rc_i = pr_rc_it ( true_set[prot], prob[prot], t )
      pr_t.append ( pr_i )
      rc_t.append ( rc_i )

    # get "f score" at each threshold
    pr_t = np.array(pr_t)
    rc_t = np.array(rc_t)
    f_value[counter] =  pr_rc_t (pr_t, rc_t)

  return np.nanmax ( f_value )



