

i = torch.LongTensor([[0, 1, 1],
                      [2, 0, 2]])

v = torch.FloatTensor([3, 4, 5])

m1 = torch.sparse.FloatTensor(i, v, torch.Size([2,3])) # .to_dense()

m2 = torch.randn(2,3)

m1.mul(m2)



c = nn.Embedding(4,5) ## 4 words dim=5
c.weight.data


