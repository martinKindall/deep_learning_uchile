import torch

def relu(T):
	if T<=0:
		return 0

	return T

print(str(relu(10)))
print(str(relu(-10)))

# print(torch.exp(3))

print(torch.is_tensor(3))

a = torch.randn(3,3,3)
# [torch.FloatTensor of size 3x3x3]
print(a)
print(torch.is_tensor(a))


