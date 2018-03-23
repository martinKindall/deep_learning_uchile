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
    def __init__(self, F, l_h, l_a, C, params=[]):
      if len(params) > 0:
        self.W_1 = params[0][0]
        self.b_1 = params[0][1]
        self.W_2 = params[1][0]
        self.b_2 = params[1][1]
        self.U = params[2][0]
        self.c_init = params[2][1]

      else:
        self.F = F
        self.l_h = l_h
        self.l_a = l_a
        self.C = C

        self.W_1 = torch.randn(F, l_h[0])
        self.b_1 = torch.zeros(1, l_h[0])

        self.W_2 = torch.randn(l_h[0], l_h[1])
        self.b_2 = torch.zeros(1, l_h[1])

        self.U = torch.randn(l_h[1], C)
        self.c_init = torch.zeros(1, C)
          
  
    def gpu(self):
      if torch.cuda.is_available():
        self.W_1 = self.W_1.cuda()
        self.b_1 = self.b_1.cuda()
        self.W_2 = self.W_2.cuda()
        self.b_2 = self.b_2.cuda()
        self.U = self.U.cuda()
        self.c_init = self.c_init.cuda()
  
    def cpu(self):
      self.W_1 = self.W_1.cpu()
      self.b_1 = self.b_1.cpu()
      self.W_2 = self.W_2.cpu()
      self.b_2 = self.b_2.cpu()
      self.U = self.U.cpu()
      self.c_init = self.c_init.cpu()
  
    def forward(self, x):
      if torch.cuda.is_available():
        x = x.cuda()
        self.gpu()   # redundante, corregir
          
      h_1 = sig(torch.mm(x, self.W_1) + self.b_1)
      h_2 = sig(torch.mm(h_1, self.W_2) + self.b_2)
      y = softmax(torch.mm(h_2, self.U) + self.c_init)
      pdb.set_trace()

      return y

red_neuronal = FFNN(4, [4,4], ['algo'], 2)
red_neuronal.forward(torch.randn(3,4))