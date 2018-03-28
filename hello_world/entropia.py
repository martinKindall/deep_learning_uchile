import torch
import pdb

def cross_ent_loss(Q,P):
	dimension = 0
	q_log = torch.log(Q)
	product = torch.mul(P, torch.reciprocal(q_log))

	pdb.set_trace()

	return torch.sum(product)/Q.size(dimension)

def main():
	a = torch.ones(3,3) * 3
	b = torch.ones(3,3) * 5

	cross_ent_loss(a, b)

main()