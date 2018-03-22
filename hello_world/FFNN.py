import pdb
import torch

class FFNN():
	def __init__(self, F, l_h, l_a, C):
		self.F = F
		self.l_h = l_h
		self.l_a = l_a
		self.C = C

		self.parametros = []
		self.W_1 = torch.randn(F, l_h)
		self.b_1 = torch.zeros(1, l_h)
		self.U = torch.randn(l_h, C)
		self.c_init = torch.zeros(1, C)

	def gpu(self):
	    if torch.cuda.is_available():
		    self.W_1 = self.W_1.cuda()
		    self.b_1 = self.b_1.cuda()
		    self.U = self.U.cuda()
		    self.c_init = self.c_init.cuda()
  
	def cpu(self):
	    self.W_1 = self.W_1.cpu()
	    self.b_1 = self.b_1.cpu()
	    self.U = self.U.cpu()
	    self.c_init = self.c_init.cpu()

red_neuronal = FFNN(10, 10, ['algo'], 2)
pdb.set_trace()