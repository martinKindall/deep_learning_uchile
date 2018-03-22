import pdb
import torch

def sig(T):
  	return torch.reciprocal(1 + torch.exp(-1 * T))	

# por ahora softmax estara implementada solo para tensores en 2-D
def softmax(T, dim=0, estable=True):
	denom_softmax = torch.div(T, 2)
	denom_softmax = torch.exp(denom_softmax)
	denom_softmax = torch.mm(denom_softmax, torch.transpose(denom_softmax, 0, 1))
	denom_softmax = torch.reciprocal(torch.diag(denom_softmax))

	return torch.mm(torch.diag(denom_softmax), T.exp())

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
		self.c_bias = torch.zeros(1, C)

	def gpu(self):
	    if torch.cuda.is_available():
		    self.W_1 = self.W_1.cuda()
		    self.b_1 = self.b_1.cuda()
		    self.U = self.U.cuda()
		    self.c_bias = self.c_bias.cuda()
  
	def cpu(self):
	    self.W_1 = self.W_1.cpu()
	    self.b_1 = self.b_1.cpu()
	    self.U = self.U.cpu()
	    self.c_bias = self.c_bias.cpu()

	def forward(self, x):
		h_1 = sig(torch.mm(x, self.W_1) + self.b_1)
		y = softmax(torch.mm(h_1, self.U) + self.c_bias)
		pdb.set_trace()

		return y

red_neuronal = FFNN(4, 4, ['algo'], 2)
red_neuronal.forward(torch.randn(3,4))