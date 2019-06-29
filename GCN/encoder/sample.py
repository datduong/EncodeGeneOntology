



from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string, re, sys, os
from tqdm import tqdm
import numpy as np
from collections import namedtuple
from tempfile import TemporaryDirectory

import logging
import json

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.init import xavier_uniform_
from torch.utils.data import DataLoader, Dataset, RandomSampler

from torch.nn import CrossEntropyLoss, MSELoss
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score


## 

from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

## do we need entire graph? how do we know indexing in a batch match with edge 

x = torch.tensor([[2,1], [5,6], [3,7], [12,0]], dtype=torch.float)
y = torch.tensor([0, 1, 0, 1], dtype=torch.float)

# the first list contains the index of the source nodes, 
# while the index of target nodes is specified in the second list.
edge_index = torch.tensor([[0, 1, 2, 0, 3],
                           [1, 0, 1, 3, 2]], dtype=torch.long)

data = Data(x=x, y=y, edge_index=edge_index)

gcn = GCNConv(2, 3)

gcn.forward(x, edge_index)

emb = nn.Embedding(10,2)
emb.weight.shape

gcn.forward(emb.weight, edge_index)


from torch_geometric.datasets import Planetoid
from torch_geometric.data import DataLoader

dataset = Planetoid(root='/tmp/Cora', name='Cora')

dataset.edge_index

loader = DataLoader(dataset, batch_size=32, shuffle=True)

for batch in loader:
  batch
  batch.num_graphs
  break 

row, col = edge_index
degree(col,4)

