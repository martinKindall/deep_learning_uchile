import torch

def relu(T):
	S = torch.abs(T)
	return torch.div(torch.add(T, 1, S), 2)

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

main2()