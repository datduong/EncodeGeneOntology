
import sys, os, pickle, re
from copy import deepcopy
import numpy as np


def GetSw (term,IcGO):
	# @IcGO is dict {GO:IC}
	if term not in IcGO:
		return 0

	ICofTerm = IcGO[term]
	if ICofTerm == 0:
		return 1

	return ( 1/( 1 + np.exp( -1/ICofTerm ) ) )

def GetSv (term,IcGO,AncestorGO):
	ancestors = deepcopy(AncestorGO [term]) ## hard copy ??
	ancestors = ancestors + [term] ## add itself
	swOfAncestors = [ GetSw(x,IcGO=IcGO) for x in ancestors ]
	return ( np.sum ( swOfAncestors ) )

def GetCommonParents (term1,term2,AncestorGO):
	set1 = set(AncestorGO[term1]+[term1]) ## ancestors including the inputs
	set2 = set(AncestorGO[term2]+[term2])
	return list ( set1.intersection(set2) )

def Sum2Sw (term1,term2,IcGO,AncestorGO):

	ancestors = GetCommonParents(term1,term2,AncestorGO)
	if len(ancestors) == 0:
		return 0

	swsum = [ GetSw (x, IcGO=IcGO) for x in list(ancestors) ]
	return ( 2 * np.sum(swsum) )

def Aic2GO (term1,term2,IcGO,AncestorGO):
	svA = GetSv(term1,IcGO,AncestorGO)
	svB = GetSv(term2,IcGO,AncestorGO)
	numerator = Sum2Sw(term1,term2,IcGO,AncestorGO)
	# if numerator == -np.inf: ## nothing is shared between 2 go terms, only the root is shared.
		# return 0 # -np.inf
	return ( numerator/(svA+svB) )

