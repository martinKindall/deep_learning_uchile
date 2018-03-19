import torch

def relu(T):
	S = torch.abs(T)
	return torch.div(torch.add(T, 1, S), 2)

def sig(T):
  	return torch.reciprocal(1 + torch.exp(-1 * T))	

def swish(T, beta):
	return torch.mul(T, sig(torch.mul(T, beta)))

def celu(T, alfa):
	positive = relu(T)
	negative = torch.mul(relu(torch.mul(T, -1)), -1)
	celu_T = torch.mul(torch.add(torch.exp(torch.div(negative, alfa)), -1), alfa)

	return torch.add(positive, 1, celu_T)

def softmax(T, dim=0, estable=True):
	print(T[dim])

def main1():

	print(str(relu(10)))
	print(str(relu(-10)))

	# print(torch.exp(3))

	# print(torch.is_tensor(3))

	print("Input tensor: \n")
	a = torch.randn(3,3,3)
	print(a)
	# [torch.FloatTensor of size 3x3x3]
	# print(torch.is_tensor(a))
	# True
	# print(torch.numel(a))
	# 27
	b = torch.abs(a)

	print("Output relu tensor: \n")
	relu = torch.div(torch.add(a, 1, b), 2)
	print(relu)

def main2():
	a = torch.randn(3,3,3)

	print("Input tensor: \n")
	print(a)
	print("Relu'd tensor: \n")
	print(relu(a))

def main3():
	a = torch.randn(3,3,3)

	print("Input tensor: \n")
	print(a)
	print("swish'd tensor: \n")
	print(swish(a, 1.5))

def main4():
	a = torch.randn(3,3,3)

	print("Input tensor: \n")
	print(a)
	print("celu'd tensor: \n")
	print(celu(a, 1.5))

def main5():
	a = torch.randn(2,2,2)

	print("Input tensor: \n")
	print(a)

	print("Input trans tensor: \n")
	print(torch.transpose(a, 0, 1))

	print("multiplying tensors")
	print(torch.matmul(a,torch.transpose(a, 0, 1)))

	# print("celu'd tensor: \n")
	# print(softmax(a))

main5()