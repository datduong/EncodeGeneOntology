


import sys, re, random, string, json, operator, time # pyLDAvis
import numpy as np


#### using f-max which is a variation of the f-measure (not exactly the standard f1-score)


def pr_rc_it (truth, prob, t):
  yhat = np.where ( prob > t ) [0] ## prediction probabilities over a threshold
  if (len(yhat) == 0) or (len(truth) == 0) : ## @truth can be [] if protein has no labels
    return 0 , 0 , len(yhat)
  #
  truth = np.where (truth > 0) [0] ## indexing of true labels
  if len(truth) == 0: ## sample with no labels
    return 0 , 0 , len(yhat)
  #
  num = len ( set (yhat).intersection ( set(truth) ) ) * 1.0
  ##! precision
  pr_i = num / len(yhat)
  ##! recall
  rc_i = num / len(truth) ##? len(truth) is same as sum_over_each go_labelset Indicator(if go is true label for this protein)
  return pr_i, rc_i, len(yhat) #! @len(yhat) is number of labels over thresshold t in this protein


def pr_rc_t (pr_t, rc_t, m_t): # $pr_t is array over prot.
  #? @pr_t will be 0, when @num=0, or len(truth)=0, or len(yhat)=0
  # mt = np.where ( pr_t > 0 ) [0] #! playing around
  mt = np.where ( m_t > 0 ) [0] #! use this one
  # print ('\ncount where proteins have non-zero prediction {}'.format(len(mt)))
  if len(mt) == 0 : # @mt is number of proteins on which at least one prediction was made above threshold t.
    pr_t = 0
  else:
    pr_t = np.mean ( pr_t [mt] )
  #
  rc_t = np.mean ( rc_t ) ##! over all protein n
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
    m_t = []
    for prot in range(true_set.shape[0]):
      pr_i , rc_i, overT_i = pr_rc_it ( true_set[prot], prob[prot], t )
      pr_t.append ( pr_i )
      rc_t.append ( rc_i )
      m_t.append ( overT_i )
    # get "f score" at each threshold
    pr_t = np.array(pr_t) #! many protein, at threshold t
    rc_t = np.array(rc_t)
    m_t = np.array(m_t)
    f_value[counter] =  pr_rc_t (pr_t, rc_t, m_t)
  return np.nanmax ( f_value ) , 0, 0


def evaluate_annotations(real_annots, pred_annots, IC_dict):
  total = 0.0
  p = 0.0
  r = 0.0
  p_total= 0.0
  ru = 0.0
  mi = 0.0
  fps = []
  fns = []
  for i in range(len(real_annots)):
    if len(real_annots[i]) == 0: ##!! skip if proteins have no labels in this ontology
      continue
    tp = set(real_annots[i]).intersection(set(pred_annots[i]))
    fp = pred_annots[i] - tp #? set operation
    fn = real_annots[i] - tp
    ## * compute Smin
    for go_id in fp:
      mi += IC_dict[go_id]
    for go_id in fn:
      ru += IC_dict[go_id]
    ## *
    fps.append(fp)
    fns.append(fn)
    tpn = len(tp) #! count true positive
    fpn = len(fp) #! count false positive
    fnn = len(fn) #! count false negative
    total += 1
    recall = tpn / (1.0 * (tpn + fnn))
    r += recall
    if len(pred_annots[i]) > 0:
      p_total += 1
      precision = tpn / (1.0 * (tpn + fpn))
      p += precision
  if total == 0: 
    return 0, 0, 0, 0, 0, 0, 0, 0
  ru /= total
  mi /= total
  r /= total
  if p_total > 0:
    p /= p_total
  f = 0.0
  if p + r > 0:
    f = 2 * p * r / (p + r)
  s = np.sqrt(ru * ru + mi * mi)
  # print ('total protein count is {}, total with valid prediction {}'.format(total,p_total))
  coverage = p_total/total
  return f, p, r, s, ru, mi, fps, fns, coverage


def f_max2 ( true_set, prob, threshold=np.arange(0.005,1,.01), IC_dict=None, label_names=None ) :
  # @true_set, @prob are np.2d-array
  # @threshold is np.vector
  label_names = np.array(label_names) # for indexing
  f_value = np.zeros( len(threshold) )
  s_min = np.zeros( len(threshold) ) * 100 ## large number
  coverage = np.zeros( len(threshold) )
  counter = -1
  real_annots = [] ##!! new threshold, we still have same true label
  for prot in range(true_set.shape[0]):
    # print (np.where( true_set[prot] > 0 ) [0])
    real_annots.append ( set ( label_names [ np.where( true_set[prot] > 0 ) [0] ].tolist() ) ) ## actual names, not number indexing
  # prob will have different set based on threshold @t
  for t in threshold :
    counter = counter + 1
    pred_annots = []
    for prot in range(true_set.shape[0]):
      pred_annots.append ( set ( label_names [ np.where( prob[prot] > t ) [0] ].tolist() ) ) #! keep index of where prediction is higher than t
    #
    output = evaluate_annotations(real_annots, pred_annots, IC_dict)
    f_value[counter] = output[0] # take first entry
    s_min[counter] = output[3]
    coverage[counter] = output[8]
  best_fmax = np.nanmax ( f_value )
  best_coverage = coverage [ np.where ( f_value == best_fmax )[0] ] ## best coverage wrt fmax
  return best_fmax , np.min(s_min), best_coverage


##! debug
# true_set = np.array ( [[0,1,1,1,0,0], [0,0,1,1,0,0], [0,1,0,0,0,0], [1,0,0,0,0,0] ] )
# prob = np.array ( [[0,0,1,1,1,1], [0,0,0,0,0,0], [0,0,0,0,1,0], [0,0,0,0,0,0] ] )
# f_max ( true_set, prob, threshold=np.array([.5]) )
# f_max2 ( true_set, prob, threshold=np.array([.5]) )


# true_set = np.array ( [[0,1,1,1,0,0], [0,0,1,1,0,0], [0,0,0,0,0,0], [1,0,0,0,0,0] ] )
# prob = np.array ( [[0,0,0,0,0,1], [0,0,0,0,0,0], [0,0,0,0,0,0], [1,0,0,0,0,0] ] )
# f_max ( true_set, prob )
# f_max2 ( true_set, prob )

