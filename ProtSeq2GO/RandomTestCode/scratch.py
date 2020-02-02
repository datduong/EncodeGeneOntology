## edge is child-->parent
AdjacencyMatrix = torch.zeros(args.num_label,args.num_label)
for i in range( args.num_label ): 
  ## given parents, fill out the 1 at the children location
  AdjacencyMatrix[i,i] = 1 ## identity 
  child = np.where(edge_index[1] == i)[0] ## if parent has children, then it will be in @edge_index[1] ## edge is child-->parent
  child = list ( set ( edge_index[0,child] ) ) ## get back the node_index_value
  if len(child) > 0: 
    AdjacencyMatrix[i,child] = 1

print ('see adjacency sum row 9565, {}'.format (torch.sum(AdjacencyMatrix[9565])) ) 


max_num_child = 467+1 # because of edge-to-self

# max_num_child = 0 ## will see 467
# for i in range( args.num_label ): 
#   child = np.where(edge_index[1] == i)[0] ## if parent has children, then it will be in @edge_index[1] ## edge is child-->parent
#   child = len ( set ( edge_index[0,child] ) ) ## get back the node_index_value
#   if child > max_num_child: 
#     max_num_child = child

AdjacencyMatrix = np.zeros((args.num_label, max_num_child)) + args.num_label ## padding index is largest node-index
for i in range( args.num_label ): 
  ## given parents, fill out the 1 at the children location
  look_up = [i] ## add itself 
  child = np.where(edge_index[1] == i)[0] ## if parent has children, then it will be in @edge_index[1] ## edge is child-->parent
  child = list ( set ( edge_index[0,child].data.numpy() ) ) ## get back the node_index_value
  if len(child) > 0: 
    look_up = look_up + child ## node-as-index to look up
  AdjacencyMatrix[ i, 0:len(look_up) ] = look_up


AdjacencyMatrix = torch.LongTensor( AdjacencyMatrix ) ## need type Long if we do nn.Embedding



self.LinearRegression = nn.ModuleList([ nn.Sequential( nn.Linear(args.go_vec_dim * 4, args.go_vec_dim) , nn.ReLU(), nn.Linear(args.go_vec_dim, 1) ) for i in range(args.num_label_to_test)])
# example 
# z = torch.nn.ModuleList([torch.nn.Linear(2, 1) for i in range(2)])
# n = torch.randn(3,2)
# [z[i](n[i]) for i in range(2)]
# pred = torch.zeros( prot_emb.shape[0], self.args.num_label_to_test).cuda() 
# for b in range (prot_emb.shape[0]): ## for each observation in batch
#   xb = [ self.LinearRegression[i] ( combined_representation[b][i] ) for i in range(self.args.num_label_to_test) ]
#   pred[b] = torch.cat (xb, dim=0) ## dim=0, because we have [ [num1], [num2] ... ]


def constraint_coef (self, coef, go_emb, true_label): 
  ## only constraint labels that are assigned ## so we need @true_label
  ## coef is linear regresion coef
  coef_loss = 0
  for index in range(true_label.shape[0]): ## each observation in this batch 

    where_one = (true_label[index]==1).nonzero().squeeze(1) ## 1D array 

    coef_i = coef[where_one] ## get coef based on indexing
    coef_i = self.Coef2Label(coef_i) ## same dim as go vector

    go_emb_i = go_emb[where_one]

    diff = (coef_i - go_emb_i).mul(coef_i - go_emb_i).sum(1) ## same as dot prot

    coef_loss = coef_loss + 0.00001 * diff.mean() ## take mean over number of true label for each obs
  
  return coef_loss



class ProtSeq2GODeepGoGcn (ProtSeq2GOBase): ## doesn't work
  ## run 1D or 2D conv then use attention layer
  def __init__(self,ProtEncoder, GOEncoder, args, **kwargs):
    super().__init__(ProtEncoder, args)

    self.maxpool_layer = maxpoolNodeLayer(kwargs['AdjacencyMatrix'], args)

    self.GOEncoder = GOEncoder
    if self.args.fix_go_emb:
      for p in self.GOEncoder.parameters():
        p.requires_grad=False

  def make_optimizer (self):
    if self.args.fix_prot_emb or self.args.fix_go_emb:
      return torch.optim.Adam ( [p for n,p in self.named_parameters () if ("ProtEncoder" not in n) or ("GOEncoder" not in n)] , lr=self.args.lr )

    elif self.args.fix_prot_emb and not self.args.fix_go_emb:
      return torch.optim.Adam ( [p for n,p in self.named_parameters () if ("ProtEncoder" not in n)] , lr=self.args.lr )

    ## if don't fix_prot_emb then we should also don't fix_go_emb
    else:
      return torch.optim.Adam ( self.parameters(), lr=self.args.lr )

  def forward( self, prot_idx, mask,  **kwargs ):
    ## @go_emb should be joint train?
    # go_emb will be computed each time. so it's costly ?

    print ('see param')

    go_emb = self.GOEncoder.gcn_2layer(kwargs['labeldesc_loader'],kwargs['edge_index'])
    for n,p in self.named_parameters () :  ## if other layers is updated
      print ("\n")
      print (n)
      print (p)

    # prot_emb is usually fast, because we don't update it ?
    prot_emb = self.maxpool_prot_emb ( prot_idx, mask )
    pred = self.match_prob ( prot_emb, go_emb )

    pred = F.sigmoid(pred) ## everything is greater than 0, so when we do max, it doesn't get negative value
    pred = self.maxpool_layer ( pred )
    return pred




class ProtSeq2GOBase (nn.Module):
  ## base class that has Prot encoder
  def __init__(self, ProtEncoder, args, **kwargs):
    super(ProtSeq2GOBase, self).__init__()

    self.args = args

    self.ProtEncoder = ProtEncoder
    if self.args.fix_prot_emb:
      for p in self.ProtEncoder.parameters(): ## save a lot of mem when turn off gradient
        p.requires_grad=False

    self.conv1d = nn.Conv1d ( args.prot_vec_dim , args.go_vec_dim, kernel_size=7, stride=1, padding=3)

    self.ReduceGoEmbProtEmb = nn.Sequential(nn.Linear(args.go_vec_dim + args.go_vec_dim, args.mid_dim),
                                            nn.Tanh())

    self.LinearRegression = nn.Linear( args.go_vec_dim + args.prot_vec_dim, args.num_label_to_test ) ## use trick

    self.classify_loss = nn.BCEWithLogitsLoss()
    # self.classify_loss = nn.BCELoss()

  def maxpool_prot_emb (self,prot_idx,mask):

    ## @mask is batch x max_len_input
    ## take maxpool of prot emb matrix
    prot = self.ProtEncoder (prot_idx) ## batch_size x num_amino x dim
    prot = F.relu ( self.conv1d ( prot.transpose(1,2) ) ) ## conv takes dim x len

    # output is batch x dim x len
    # masking will be needed
    mask = mask.unsqueeze(1)
    mask = (1.0 - mask) * -10000 # when take softmax or maxpool, these -10000 will be 0 or doesn't count
    prot = prot + mask

    prot , _ = torch.max ( prot, 2 )
    return prot

  def match_prob (self, prot_emb, go_emb ) :

    ## for each prot, expand to same size as numGO, concat, do forward. ... no way around this ?
    pred = torch.zeros( prot_emb.shape[0], self.args.num_label_to_test ).cuda() ## dummy holder

    for i in range ( prot_emb.shape[0] ):
      # @go_emb is 2D, so concat on dim=1 (the numb. of go term dim)
      # use @prot_emb.shape[1] because we can transform prot-vec

      x = torch.cat ( (go_emb, prot_emb[i].expand( self.args.num_label_to_test, prot_emb.shape[1] )) , dim=1 ) # @x is num_go_label x some_dim
      x = self.ReduceGoEmbProtEmb(x)

      # we do linear regression (this is the same as Dense(1) in keras deepgo. for each GO term, we have a single linear regression weight. take sigmoid later)
      pred[i] = self.LinearRegression.weight.mul(x).sum(1) + self.LinearRegression.bias ## dot-product sum up

    return pred ## if print @pred, we will see grad_fn=<CopySlices>

  def forward( self, prot_idx, mask, **kwargs ): ## @go_emb should be joint train?
    # prot_emb = self.maxpool_prot_emb ( prot_idx, mask)
    prot_emb = self.ProtEncoder ( prot_idx, **kwargs )
    pred = self.match_prob ( prot_emb, kwargs['go_emb'] )
    return pred



import pandas as pd 

df = pd.read_csv( "/u/flashscratch/d/datduong/deepgo/data/train/train-mf.tsv" , sep="\t")

len_seq = []
for seq in list(df['Sequence'])  :
  len_seq.append ( len(seq) ) 

  


 def concat_compare (self, prot_emb, go_emb):
    ## take 1 prot append/compare
    ## if we fix @go_emb, then we can do this in batch mode pretty fast

    prot_emb_reduce = self.ReduceProtEmb (prot_emb) ## make into same dim as go vector

    prot_emb_reduce = prot_emb_reduce.unsqueeze(1) ## make into 3D : batch x 1 x dim, able to do computation using broadcast

    pred = prot_emb_reduce.mul ( go_emb ).sum(2) + self.LinearRegression.bias

    ## this is pointwise, batch x num_go x dim... here, 1 obs in batch = 1 prot
    # angle = torch.mul ( prot_emb_reduce, go_emb )
    # distance = torch.abs (prot_emb_reduce - go_emb) 

    # # prot_emb_reduce, _ = torch.broadcast_tensors ( prot_emb_reduce, go_emb ) ## output batch x num_go x dim

    # ## @prot_emb is 2D, batch x dim , so we make it into batch x num_go x dim
    # prot_emb3d = prot_emb.unsqueeze(1).expand( prot_emb.shape[0], self.args.num_label_to_test, self.final_prot_dim )

    # prot_emb3d = torch.cat((prot_emb3d, distance), dim=2) # append maxpool prot_emb with abs(prot_emb_reduce - go_emb)

    # # pred = self.LinearRegression ( combine_vec ).squeeze(2) ## use @.squeeze(2) to make into 2D, so we have batch x num_go

    # pred = self.LinearRegression.weight.mul(prot_emb3d).sum(2) + self.LinearRegression.bias ## dot-product sum up

    return pred



## do concat/compare
## testing on subset so we use kwargs['label_to_test_index']
prot_go_vec = self.concat_prot_go(prot_emb,go_emb[kwargs['label_to_test_index']]) ## batch x num_go x dim

## append ppi-network
prot_interact_emb = prot_interact_emb.unsqueeze(1) ## make into 3D : batch x 1 x dim
prot_interact_emb = prot_interact_emb.expand( prot_interact_emb.shape[0], self.args.num_label_to_test, self.args.prot_interact_vec_dim ) ## batch x num_go x dim. if we do prot_interact_emb[0] we get back 1st protein vec in ppi network, duplicated many times to match num_go

## highway network
prot_emb = prot_emb.unsqueeze(1) ## make into 3D : batch x 1 x dim
prot_emb = prot_emb.expand( prot_emb.shape[0], self.args.num_label_to_test, self.args.prot_vec_dim )

prot_emb_concat = torch.cat ( (prot_emb , prot_go_vec, prot_interact_emb) , dim=2 ) ## append to known ppi-network

prot_emb, go_emb = torch.broadcast_tensors ( prot_emb, go_emb ) ## output batch x num_go x dim ## works only if same dim 300 and 300 ?? 
