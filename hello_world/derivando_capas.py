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

dimL = 15

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

# Acá tu trozo de código. 
# Primero agregamos algunas variables dummy para chequear 
# que al menos las dimensiones están correctas

dimk = 20
dimkm1 = 15

hk = torch.ones(B,dimk)
Wk = torch.ones(dimk,dimkm1)
bk = torch.ones(dimkm1)

uk = hk.mm(Wk).add(bk)

dL_dhkm1 = torch.ones(B,dimkm1)

# Ahora tu fórmula para el gradiente.
# Esto debes repetirlo para relu, celu, y swish

# para sigmoid sigmoid

dL_duk = torch.mul(dL_dhL, torch.mul(hL, torch.mul(torch.add(hL, -1), -1)))
dL_dWk = None
dL_dbk = None
dL_dhk = None

# El gradiente debe coincidir en dimensiones con las variables

assert dL_duk.size() == uk.size()
# assert dL_dhk.size() == hk.size()
# assert dL_dbk.size() == bk.size()
# assert dL_dWk.size() == Wk.size()