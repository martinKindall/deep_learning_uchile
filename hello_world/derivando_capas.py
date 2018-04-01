import torch
import pdb

# Para ir chequeando que al menos las dimensiones de los tensores son 
# consistentes usaremos las varibles *dummy* a continuación.

B = 5; C = 10

y = torch.ones(B,C)
y_pred = torch.ones(B,C)

# Acá tu trozo de código. 
# Primero agregamos algunas variables dummy para chequear 
# que al menos las dimensiones están correctas

dimL = 40

hL = torch.ones(B,dimL)
U = torch.ones(dimL,C)
c_bias = torch.ones(C)

uL = hL.mm(U).add(c_bias)

# Notamos que por regla de la cadena, 
# dL_duL = dL_dypred * dypred_duL
# el primer termino de la derecha es la derivada
# de cross entropy 
# el segundo termino de la derecha es la derivada de softmax

#pdb.set_trace()

dL_duL = torch.mul(torch.mul(y, torch.add(y_pred, -1)), 1/y_pred.size(0))

# El gradiente debe coincidir en dimensiones con la variable

assert dL_duL.size() == uL.size()


dL_dU = torch.mm(torch.transpose(hL, 0, 1), dL_duL)

dL_dc = torch.mm(torch.ones(1, dL_duL.size(0)), dL_duL)

dL_dhL = torch.mm(dL_duL, torch.transpose(U, 0, 1))

# El gradiente debe coincidir en dimensiones con las variables

assert dL_dU.size() == U.size()
assert dL_dc.size(1) == c_bias.size(0)
assert dL_dhL.size() == hL.size()