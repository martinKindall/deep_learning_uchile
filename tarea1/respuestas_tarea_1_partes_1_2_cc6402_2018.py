# -*- coding: utf-8 -*-
"""Respuestas_Tarea_1_Partes_1_2_CC6402_2018.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1eLpUjYXIdvWatSxUH8oEliNLXpGEj927

# Tarea 1 <br/> CC6204 Deep Learning, Universidad de Chile  <br/> Hoja de respuestas partes 1 y 2 
## Nombre: Martin Cornejo Saavedra
Fecha sugerida para completar esta parte: 23 de marzo de 2018
"""

# instalacion de los paquetes necesarios

is_colab = False

if (is_colab):
    from os import path
    from wheel.pep425tags import get_abbr_impl, get_impl_ver, get_abi_tag
    platform = '{}{}-{}'.format(get_abbr_impl(), get_impl_ver(), get_abi_tag())

    accelerator = 'cu80'

    #!pip install -q http://download.pytorch.org/whl/{accelerator}/torch-0.3.0.post4-{platform}-linux_x86_64.whl torchvision
    #!pip install -q ipdb

import torch
import pdb

"""# Parte 1: Funciones de activación, derivadas y función de salida

## 1a) Funciones de activación
"""

def relu(T):
    T[T < 0] = 0
    return T

def sig(T):
    return torch.reciprocal(1 + torch.exp(-1 * T))

def swish(T, beta):
    return torch.mul(T, sig(torch.mul(T, beta)))

def celu(T, alfa):
    positive = relu(T)
    negative = torch.mul(relu(torch.mul(T, -1)), -1)
    celu_T = torch.mul(torch.add(torch.exp(torch.div(negative, alfa)), -1), alfa)

    return torch.add(positive, 1, celu_T)

"""## 1b) Derivando las funciones de activación

\begin{equation}
\frac{\partial\ \text{relu}(x)}{\partial x} =
\left\{
	\begin{array}{ll}
		1  & \mbox{si } x \geq 0 \\
		0  & \mbox{~} 
	\end{array}
\right. 
\end{equation}
<br>

Dado $ \sigma (x) = sigmoid(x)$, tenemos que:

\begin{eqnarray}
\frac{\partial\ \text{swish}(x, \beta)}{\partial x} & = \sigma (\beta x) + \beta x \cdot \sigma (\beta x)(1-\sigma (\beta x)) \\
& = \sigma (\beta x) + \beta x \cdot \sigma (\beta x) - \beta x \cdot \sigma (\beta x)^{2}  \\
&= \beta \cdot swish(x, \beta) + \sigma (\beta x)(1 - \beta \cdot swish(x, \beta))\\
\\
\frac{\partial\ \text{swish}(x, \beta)}{\partial \beta} & =  
x^2 \sigma (\beta x)(1 - \sigma (\beta x))\\
\end{eqnarray}
<br><br>

\begin{eqnarray}
\frac{\partial\ \text{celu}(x, \alpha)}{\partial x} & =  
\left\{
	\begin{array}{ll}
		1  & \mbox{si } x \geq 0 \\
		exp (\frac{x}{\alpha})  & \mbox{~} 
	\end{array}
\right. \\
\\
\frac{\partial\ \text{celu}(x, \alpha)}{\partial \alpha} & = 
\left\{
	\begin{array}{ll}
		0  & \mbox{si } x \geq 0 \\
		exp (\frac{x}{\alpha})(1 - \frac{x}{\alpha}) - 1  & \mbox{~} 
	\end{array}
\right. \\
\end{eqnarray}

## 1c) Softmax

Dada la funcion `softmax` sabemos que cada elemento de la secuencia $\text{softmax}(x_1,\ldots,x_n)$ tiene la forma

\begin{equation}
s_i = \frac{e^{x_i}}{\sum_{j=1}^{n}e^{x_j}}
\end{equation}

Luego, para cada elemento de la secuencia $\text{softmax}(x_1-M,\ldots,x_n-M)$ se tiene

\begin{equation}
s_i = \frac{e^{x_i-M}}{\sum_{j=1}^{n}e^{x_j-M}} = \frac{e^{-M}e^{x_i}}{\sum_{j=1}^{n}e^{-M}e^{x_j}} = \frac{e^{-M}e^{x_i}}{e^{-M}\sum_{j=1}^{n}e^{x_j}} = \frac{e^{x_i}}{\sum_{j=1}^{n}e^{x_j}}
\end{equation}

Demostrando que $\text{softmax}(x_1-M,\ldots,x_n-M) = \text{softmax}(x_1,\ldots,x_n)$.
"""

# por ahora softmax estara implementada solo para tensores en 2-D
def softmax(T, dim=0, estable=True):
    denom_softmax = torch.div(T, 2)
    denom_softmax = torch.exp(denom_softmax)
    denom_softmax = torch.mm(denom_softmax, torch.transpose(denom_softmax, 0, 1))
    denom_softmax = torch.reciprocal(torch.diag(denom_softmax))

    return torch.mm(torch.diag(denom_softmax), T.exp())

"""# Parte 2: Red neuronal y pasada hacia adelante (forward)

## 2a) Clase para red neuronal, 2b) Usando la GPU, 2c) Pasada hacia adelante
"""

class FFNN():
    def __init__(self, F, l_h, l_a, C):
        self.F = F
        self.l_h = l_h
        self.l_a = l_a
        self.C = C

        self.W_1 = torch.randn(F, l_h)
        self.b_1 = torch.zeros(1, l_h)
        self.U = torch.randn(l_h, C)
        self.c_init = torch.zeros(1, C)
  
    def gpu(self):
        if torch.cuda.is_available():
            self.W_1 = self.W_1.cuda()
            self.b_1 = self.b_1.cuda()
            self.U = self.U.cuda()
            self.c_init = self.c_init.cuda()
  
    def cpu(self):
        self.W_1 = self.W_1.cpu()
        self.b_1 = self.b_1.cpu()
        self.U = self.U.cpu()
        self.c_init = self.c_init.cpu()
  
    def forward(self, x):
        if torch.cuda.is_available():
          x = x.cuda()
          self.gpu()   # redundante, corregir
          
        h_1 = sig(torch.mm(x, self.W_1) + self.b_1)
        y = softmax(torch.mm(h_1, self.U) + self.c_init)
        pdb.set_trace()

        return y

"""## 2d) Probando tu red con un modelo pre-entrenado"""

red_neuronal = FFNN(4, 4, ['algo'], 2)
red_neuronal.forward(torch.randn(3,4))

# Tu código visualizando los ejemplos incorrectos acá