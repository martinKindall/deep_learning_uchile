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
c = torch.ones(C)

uL = hL.mm(U).add(c)

# Notamos que por regla de la cadena, 
# dL_duL = dL_dypred * dypred_duL
# el primer termino de la derecha es la derivada
# de cross entropy 
# el segundo termino de la derecha es la derivada de softmax

dL_dypred = torch.mul(torch.mul(y, torch.reciprocal(y_pred)), -1/y_pred.size(0))

pdb.set_trace()

#dL_duL = 

# El gradiente debe coincidir en dimensiones con la variable

assert dL_duL.size() == uL.size()