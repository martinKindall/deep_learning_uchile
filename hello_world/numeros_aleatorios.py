import torch
import pdb
import numpy as np

N = 20 # numero de ejemplos
f = 7 # numero de features

X = torch.rand(N,f)
X = torch.bernoulli(X)

Y = torch.rand(N,1)
Y = torch.bernoulli(Y)

pdb.set_trace()