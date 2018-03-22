import pdb
import torch

class FFNN():
	def __init__(self, F, l_h, l_a, C):
		self.F = F
		self.l_h = l_h
		self.l_a = l_a
		self.C = C

		self.W_1 = torch.randn(F, l_h)
		self.b_1 = torch.zeros(1, l_h)
		self.U = torch.randn(l_h, C)
		self.c_init = torch.zeros(1, C)

red_neuronal = FFNN(10, 10, ['algo'], 2)
pdb.set_trace()