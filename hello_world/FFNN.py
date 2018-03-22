import pdb

class FFNN():
	def __init__(self, F, l_h, l_a, C):
		self.F = F
		self.l_h = l_h
		self.l_a = l_a
		self.C = C

red_neuronal = FFNN(10, [10], ['algo'], 2)
pdb.set_trace()
