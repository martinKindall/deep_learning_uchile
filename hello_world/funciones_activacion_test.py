import torch

def relu(T):
	if T<=0:
		return 0

	return T

print(str(relu(10)))
print(str(relu(-10)))

print(torch.exp(3))