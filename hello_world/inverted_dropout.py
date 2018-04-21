import torch
import pdb

def main():
	a = torch.ones(5,5)
	p = 0.7
	# b = (torch.randn(a.size(0), a.size(1)).double() < p)
	b = (torch.rand(a.size(0), a.size(1)).double() < p).double() * (1/p)
	print(b)
	return

main()