
import torch

z = torch.randn(3,4,5)
m = torch.FloatTensor ([[1,1,0,0],[1,1,1,0],[1,0,0,0]])

z[ m==0 ] = 50

m = m.unsqueeze(1).transpose(1,2)
# z = z.transpose(1,2) 

m * z
