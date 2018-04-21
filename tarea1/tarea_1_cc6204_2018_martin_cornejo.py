# -*- coding: utf-8 -*-
"""Tarea_1_CC6204_2018_Martin_Cornejo

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1JaaeyGEj_OShRDAoBCGtNE3k2yd1y5j5

# Tarea 1 <br/> CC6204 Deep Learning, Universidad de Chile <br/> Hoja de respuestas 

## Nombre: Martín Cornejo Saavedra
Fecha para completar la tarea: miercoles 25 de abril de 2018

## Insturcciones:
En este notebook debes dejar todo tu código de las partes 1 hasta 8 de la tarea. Debes dejar el código (y todo lo adicional que hayas programado) en las celdas designadas para ello. Las partes que se hacen a mano (o con fórmulas), puedes entregarlas al final o en un archivo separado. En la siguiente celda encontrarás un *check list* de las partes de la tarea. En cada item indica si lo completaste o no en tu entrega ('SI', o 'NO'). Por favor, no marques como 'SI' partes que no hiciste. Adicionalemnte, el lugar en tu código donde completaste cada parte, márcala con un comentario como el siguiente

```
#### Parte 4a) Método backward
```

Si bien para algunas partes puede no estar exactamente claro dónde comienza (y termina) cada parte, usa tu criterio para los comentarios. Esto es sólo para ayudarnos a corregir. Nota que las celdas de más abajo no necesariamente siguen el orden del check list. Por ejemplo en la celda donde defines tu clase para la red neuronal se espera agregues código de las partes 2, 3, 7 etc. Todo eso está en una única celda (lo entenderás cuando mires más abajo).

Por favor sigue este formato. **Si no sigues el formato te descontaremos puntaje.**

---

## Checklist

Parte|Completado
---:|---:
**1) Activación, derivadas, y salida** |
1a) Funciones de activación | SI
1b) Derivando las funciones de activación | SI
1c) Softmax | SI
**2) Red neuronal y pasada hacia adelante** |
2a) Clase para red neuronal | SI (solo 2 capas fijas, falta hacerlo para N capas)
2b) Usando la GPU | SI
2c) Pasada hacia adelante | SI
2d) Probando tu red con un modelo pre-entrenado | SI (falta accurracy)
**3) Más derivadas y backpropagation** |
3a) Entropía cruzada | SI
3b) Derivando la última capa | SI (falta desarrollo de derivar softmax)
3c) Derivando desde las capas escondidas | SI (falta derivar celu, relu, etc..)
**4) Backpropagation en nuestra red** |
4a) Método backward | SI (falta hacerlo para N capas)
4b) Checkeo de gradiente | NO
4c) Opcional: incluyendo los parámetros de celu y swish | NO
**5) Descenso de gradiente y entrenamiento** |
5a) Descenso de gradiente | SI
5b) Datos para carga | SI
5c) Entrenando la red | SI
5d) Graficando la pérdida en el tiempo | SI
5e) Entrenando con datos no random | NO
**6) Regularización** |
6a) Regularización por penalización de norma | Falta implementar el nuevo loss, para eso falta red de capa dinamica
6b) Regularización por dropout | NO
**7) Optimización** |
7a) Inicialización de Xavier | NO
7b) Descenso de gradiente con momentum | NO
7c) RMSProp | NO
7d) Adam | NO
7e) Opcional: batch normalization | NO
**8) Entrenando sobre MNIST** |
8a) Cargando y visualizando datos de MNIST | NO
8b) Red neuronal para MNIST | NO
8c) Opcional: visualización de entrenamiento y convergencia | NO

---
"""

# Este notebook está pensado para correr en CoLaboratory. 
# Comenzamos instalando las librerías necesarias.
# Si lo estás ejecutando localmente posiblemente no sea necesario
# reinstalar todo.

import os
import glob
import torch
import numpy
import pdb

# Agrega acá todo lo que quieras importar o instalar

"""## Funciones de activación, predicción y pérdida"""

def sig(T):
  return torch.reciprocal(1 + torch.exp(-1 * T))

def tanh(T):
  E = torch.exp(T)
  e = torch.exp(-1 * T)
  return (E - e) * torch.reciprocal(E + e)

#### Parte 1a) Funciones de activación

def relu(T):
    T[T < 0] = 0
    return T

def swish(T, beta):
    return torch.mul(T, sig(torch.mul(T, beta)))

def celu(T, alfa):
    positive = relu(T)
    negative = torch.mul(relu(torch.mul(T, -1)), -1)
    celu_T = torch.mul(torch.add(torch.exp(torch.div(negative, alfa)), -1), alfa)

    return torch.add(positive, 1, celu_T)

#### Parte 1c) Softmax está implementada solo para tensores en 2-D
def softmax(T, dim=0, estable=True):
    denom_softmax = torch.div(T, 2)
    denom_softmax = torch.exp(denom_softmax)
    denom_softmax = torch.mm(denom_softmax, torch.transpose(denom_softmax, 0, 1))
    denom_softmax = torch.reciprocal(torch.diag(denom_softmax))

    return torch.mm(torch.diag(denom_softmax), T.exp())

#### Parte 3a) Entropía cruzada
def cross_ent_loss(Q,P):
    dimension = 0
    q_log = torch.log(Q)
    product = torch.mul(P, torch.reciprocal(q_log))

    return -torch.sum(product)/Q.size(dimension)

"""## Clase FFNN, incialización, forward, backward y regularización"""

#### Parte 2a) Clase para red neuronal
class FFNN():
  
  def __init__(
  #    self, F, l_h, l_a, C, 
  #    wc_par=None, 
  #    keep_prob=None, 
  #    init=None,
  #    bn=None
  #):
    self, F, l_h, l_a, C, params=[],
    wc_par=None,
    keep_prob=[]
  ):
      if (len(params) > 0):
        self.W_1 = params[0][0]
        self.b_1 = params[0][1]
        self.W_2 = params[1][0]
        self.b_2 = params[1][1]
        self.U = params[2][0]
        self.c_init = params[2][1]
        
        self.c_init = self.c_init.view(1, self.c_init.size(0))
        self.b_2 = self.b_2.view(1, self.b_2.size(0))
        self.b_1 = self.b_1.view(1, self.b_1.size(0))
        
        self.l_a = l_a

      else:
        self.F = F
        self.l_h = l_h
        self.l_a = l_a
        self.C = C

        self.W_1 = torch.randn(F, l_h[0])
        self.b_1 = torch.zeros(1, l_h[0])

        self.W_2 = torch.randn(l_h[0], l_h[1])
        self.b_2 = torch.zeros(1, l_h[1])

        self.U = torch.randn(l_h[1], C)
        self.c_init = torch.zeros(1, C)
        
      self.wc_par = wc_par        
      self.keep_prob = keep_prob
        
        
  #### Parte 2b) Usando la GPU
  def gpu(self):
    if torch.cuda.is_available():
      self.W_1 = self.W_1.double()
      self.b_1 = self.b_1.double()
      self.W_2 = self.W_2.double()
      self.b_2 = self.b_2.double()
      self.U = self.U.double()
      self.c_init = self.c_init.double()
  
  def cpu(self):
    self.W_1 = self.W_1.cpu().double()
    self.b_1 = self.b_1.cpu().double()
    self.W_2 = self.W_2.cpu().double()
    self.b_2 = self.b_2.cpu().double()
    self.U = self.U.cpu().double()
    self.c_init = self.c_init.cpu().double()
    
    #### Parte 7a) Inicialización de Xavier
    #### Parte 7e) Opcional: batch normalization
    
  
  #### Parte 2c) Pasada hacia adelante
  def forward(self, x, predict=False):
    x = x.double()

    if torch.cuda.is_available():
      x = x
      self.gpu()   # redundante, corregir
      
    else:   
      self.cpu()   # redundante, corregir
      
    if (len(self.keep_prob) == 0 or not predict):
      prob_array = torch.ones(3)   # no se apagan neuronas
      
    # iterar para crear las mascaras de bits segun los largos de las matrices,
    # hacerlo dinamico segun el numero de capas
    
    self.h_1 = self.l_a[0](torch.mm(x, self.W_1) + self.b_1)
    self.h_2 = self.l_a[1](torch.mm(self.h_1, self.W_2) + self.b_2)
    y = softmax(torch.mm(self.h_2, self.U) + self.c_init)

    return y

  
  #### Parte 4a) Método backward
  def backward(self,x,y,y_pred):
    gradientes = {}
      
    # gradientes capa de salida
    
    dimL = self.b_2.size(0)
    uL = self.h_2.mm(self.U).add(self.c_init)
      
    #dL_duL = torch.mul(torch.mul(y, torch.add(y_pred, -1)), 1/y_pred.size(0))
       
    dL_duL = torch.mul(torch.add(y_pred, torch.mul(y, -1)), 1/y_pred.size(0))
    
    dL_dU = torch.mm(torch.transpose(self.h_2, 0, 1), dL_duL)
    dL_dc = torch.mm(torch.ones(1, dL_duL.size(0)).double(), dL_duL)
    dL_dh2 = torch.mm(dL_duL, torch.transpose(self.U, 0, 1))
      
    assert dL_duL.size() == uL.size()
    assert dL_dU.size() == self.U.size()
    
    #pdb.set_trace()
    
    assert dL_dc.size(1) == self.c_init.size(1)
    assert dL_dh2.size() == self.h_2.size()

    # gradientes segunda capa escondida
      
    u2 = self.h_1.mm(self.W_2).add(self.b_2)  

    # para sigmoid
    dL_du2_sig = torch.mul(dL_dh2, torch.mul(self.h_2, torch.mul(torch.add(self.h_2, -1), -1)))
  
    # para relu
    dL_duk_rel = None

    # para celu
    dL_duk_celu = None

    # para swish
    dL_duk_swish = None

    # se elige una funcion de activacion
    dL_du2 = dL_du2_sig

    dL_dW2 = torch.mm(torch.transpose(self.h_1, 0, 1), dL_du2)
    dL_db2 = torch.mm(torch.ones(1, dL_du2.size(0)).double(), dL_du2)
    dL_dh1 = torch.mm(dL_du2, torch.transpose(self.W_2, 0, 1))

    assert dL_du2.size() == u2.size()
    assert dL_dW2.size() == self.W_2.size()
    assert dL_db2.size(1) == self.b_2.size(1)
    assert dL_dh1.size() == self.h_1.size()
      
    # gradientes primera capa escondida
      
    u1 = x.mm(self.W_1).add(self.b_1)  

    # para sigmoid
    dL_du1_sig = torch.mul(dL_dh1, torch.mul(self.h_1, torch.mul(torch.add(self.h_1, -1), -1)))
  
    # para relu
    dL_duk_rel = None

    # para celu
    dL_duk_celu = None

    # para swish
    dL_duk_swish = None

    dL_du1 = dL_du1_sig

    dL_dW1 = torch.mm(torch.transpose(x, 0, 1), dL_du1)
    dL_db1 = torch.mm(torch.ones(1, dL_du1.size(0)).double(), dL_du1)
    dL_dx = torch.mm(dL_du1, torch.transpose(self.W_1, 0, 1))

    assert dL_du1.size() == u1.size()
    assert dL_dW1.size() == self.W_1.size()
    assert dL_db1.size(1) == self.b_1.size(1)
    assert dL_dx.size() == x.size()
      
    for gradiente in ['dL_db1', 'dL_dW1', 'dL_db2', 'dL_dW2', 'dL_dc', 'dL_dU']:
      gradientes[gradiente] = eval(gradiente)
      
      self.gradientes = gradientes
      
      
  def actualizarParams(self, lr):
    
    # mover esto al optimizador, para luego poder calcular los nuevos gradientes usando
    # wc_par   
    
    const = 1
    
    # Parte 6a) agregando weight decay
    if (self.wc_par != None):
      const = const - lr * self.wc_par
    
    self.W_1 = self.W_1 * const  - lr * self.gradientes['dL_dW1']
    self.b_1 = self.b_1 - lr * self.gradientes['dL_db1']

    self.W_2 = self.W_2 * const - lr * self.gradientes['dL_dW2']
    self.b_2 = self.b_2 - lr * self.gradientes['dL_db2']

    self.U = self.U * const - lr * self.gradientes['dL_dU']
    self.c_init = self.c_init - lr * self.gradientes['dL_dc']

"""## Probando tu red con un modelo pre-entrenado y visualizando casos incorrectos"""

#### Parte 2d) Probando tu red con un modelo pre-entrenado


"""## Descenso de gradiente, momentum, RMSProp y Adam"""

#### Parte 5a) Descenso de gradiente
class SGD():
  def __init__(self, red, lr):
    self.red = red
    self.lr = lr
  
  def step(self):
    self.red.actualizarParams(self.lr)
    
#### Parte 7b) Descenso de gradiente con momentum
class SGD2():
  def __init__(self, red, lr, momentum=0.9):
    pass
  
  def step():
    pass

#### Parte 7c) RMSProp
class RMSProp():
  def __init__(self, red, lr=0.001, beta=0.9, epsilon=1e-8):
    pass
  
  def step():
    pass

#### Parte 7d) Adam
class Adam():
  def __init__(self, red, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
    pass
  
  def step():
    pass

"""## Datos random para carga"""

#### Parte 5b) Datos para carga
class RandomDataset():
  def __init__(self, N, F, C):
    x = torch.rand(N, F)
    self.x = torch.bernoulli(x).double()
    self.y = torch.from_numpy(numpy.eye(C)[numpy.random.choice(C, N)]).double()
    self.large = N
  
  def __len__(self):
    return self.large
  
  def __getitem__(self, i):
    return (self.x[i,:], self.y[i,:])
  
  def paquetes(self, B):
       
    if not hasattr(self, 'arr_paquetes'):
      n_iters = int(self.x.size(0)/B)
      arr_paquetes = []  
   
      for index in range(n_iters):
        arr_paquetes.append(self.elige_batch(self.x,self.y,B))
              
      self.arr_paquetes = arr_paquetes
      
    return self.arr_paquetes
    
  # Para elegir el siguiente batch (uno al azar) desde los datos de entrada
  def elige_batch(self, X, Y, b):
    N = X.size()[0]
    x_lista = []
    y_lista = []
  
    for _ in range(b):
      i = numpy.random.randint(N)
      x_lista.append(X[i:i+1])
      y_lista.append(Y[i:i+1])      
  
    x = torch.cat(x_lista, dim=0)
    y = torch.cat(y_lista, dim=0)
    #pdb.set_trace()
  
    return x,y

"""## Loop de entrenamiento"""

#### Parte 5c) Entrenando la red
def entrenar_FFNN(red_neuronal, dataset, optimizador, epochs, B):
  
  perdidas = []
  
  for e in range(1,epochs+1):
    for x,y in dataset.paquetes(B):
      y_pred = red_neuronal.forward(x)
      #pdb.set_trace()
      loss = cross_ent_loss(y_pred,y)
      
      #6a) Regularización por penalización de norma
      if (red_neuronal.wc_par != None):
        loss += penalizacionNorma(red_neuronal.params, B)
            
      perdidas.append(loss)
      
      red_neuronal.backward(x, y, y_pred)
      optimizador.step()  
    
  return red_neuronal, perdidas  

def penalizacionNorma(arrayMatrix, B):
  return 0

"""## Entrenando con datos random y graficando la pérdida"""

#### Parte 5d) Graficando la pérdida en el tiempo

#red_entrenada, perdida = entrenar_FFNN(red, dataset, optimizador, 20, 4)

#red_neuronal = FFNN(0, [], ['algo'], 10, params)

features = 10
clases = 3
dataset = RandomDataset(1000,features,clases)

redes = [
    FFNN(features, [5, 5], [sig, sig], clases),
    FFNN(features, [15, 15], [sig, sig], clases),
    FFNN(features, [50, 50], [sig, sig], clases)
]

for idx, red_neuronal in enumerate(redes):

  red_neuronal.cpu()

  optimizador = SGD(red_neuronal, 0.001) 

  epochs = 30
  batch = 10
  red_neuronal, perdidas = entrenar_FFNN(red_neuronal, dataset, optimizador, epochs, batch)

  #pdb.set_trace()
  
True

"""## Entrenando con datos de varita mágica"""

#### Parte 5e) Entrenando con datos no random

"""## Cargando datos de MNIST"""

#### Parte 8a) Cargando y visualizando datos de MNIST

"""## Red neuronal para MNIST"""

#### Parte 8b) Red neuronal para MNIST

#### Parte 8c) Visualización de entrenamiento y convergencia

"""---

## Apéndice: partes a mano

### Parte 1b) Derivando las funciones de activación

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

### Parte 1c) Softmax

Dada la funcion `softmax` sabemos que cada elemento de la secuencia $\text{softmax}(x_1,\ldots,x_n)$ tiene la forma

\begin{equation}
s_i = \frac{e^{x_i}}{\sum_{j=1}^{n}e^{x_j}}
\end{equation}

Luego, para cada elemento de la secuencia $\text{softmax}(x_1-M,\ldots,x_n-M)$ se tiene

\begin{equation}
s_i = \frac{e^{x_i-M}}{\sum_{j=1}^{n}e^{x_j-M}} = \frac{e^{-M}e^{x_i}}{\sum_{j=1}^{n}e^{-M}e^{x_j}} = \frac{e^{-M}e^{x_i}}{e^{-M}\sum_{j=1}^{n}e^{x_j}} = \frac{e^{x_i}}{\sum_{j=1}^{n}e^{x_j}}
\end{equation}

Demostrando que $\text{softmax}(x_1-M,\ldots,x_n-M) = \text{softmax}(x_1,\ldots,x_n)$.

### Parte 3b) Derviando la última capa

\begin{equation}
\frac{\partial \cal L}{\partial u^{(L)}} = \frac{\partial \cal L}{\partial ŷ} \cdot \frac{\partial ŷ}{\partial u^{(L)}} \\
\end{equation}

Usando la notación de Einstein tenemos que:

\begin{eqnarray}
(\frac{\partial\cal L}{\partial U})_{ij} &= \frac{\partial\cal L}{\partial u^{(L)}_{kl}} \frac{\partial u^{(L)}_{kl}}{\partial U_{ij}} \\
\end{eqnarray}

Luego,

\begin{equation}
\frac{\partial u^{(L)}_{kl}}{\partial U_{ij}} = \frac{\partial (h^{(L)}_{kr}U_{rl} + c_{l})}{\partial U_{ij}} = \left\{
    \begin{array}{}
		h^{(L)}_{ki}  & \mbox{si } r = i,\ l = j \\
		0  & \mbox{~}
    \end{array}
\right.
\end{equation}

Entonces,

\begin{eqnarray}
(\frac{\partial\cal L}{\partial U})_{ij} &= \frac{\partial\cal L}{\partial u^{(L)}_{kl}} h^{(L)}_{ki} = \frac{\partial\cal L}{\partial u^{(L)}_{kj}} h^{(L)}_{ki} \\
\\
& \boxed{ \frac{\partial\cal L}{\partial U} = (h^{(L)})^{T} \frac{\partial\cal L}{\partial u^{(L)}} }
\end{eqnarray}

Análogamente

\begin{equation}
\frac{\partial u^{(L)}_{kl}}{\partial c_{i}} = \frac{\partial (h^{(L)}_{kr}U_{rl} + c_{l})}{\partial c_{i}} = \left\{
    \begin{array}{}
		1  & \mbox{si } l = j \\
		0  & \mbox{~}
    \end{array}
\right.
\end{equation}

\begin{eqnarray}
(\frac{\partial\cal L}{\partial c})_{i} &= \frac{\partial\cal L}{\partial u^{(L)}_{ki}} \cdot 1
\end{eqnarray}

\begin{equation}
\boxed{ \frac{\partial\cal L}{\partial c} = [1 \ldots 1] \frac{\partial\cal L}{\partial u^{(L)}} }
\end{equation}

Donde $[1 \ldots 1]$ es un vector de unos de largo correspondiente al numero de fila del resultado de $\frac{\partial\cal L}{\partial u^{(L)}}$ 

Finalmente,

\begin{equation}
\frac{\partial u^{(L)}_{kl}}{\partial h^{(L)}_{ij}} = \frac{\partial (h^{(L)}_{kr}U_{rl} + c_{l})}{\partial h^{(L)}_{ij}} = \left\{
    \begin{array}{}
		U_{jl}  & \mbox{si } k = i,\ r = j \\
		0  & \mbox{~}
    \end{array}
\right.
\end{equation}

\begin{eqnarray}
(\frac{\partial\cal L}{\partial h^{(L)}})_{ij} &= \frac{\partial\cal L}{\partial u^{(L)}_{kl}} U_{jl} = \frac{\partial\cal L}{\partial u^{(L)}_{il}} U_{lj}^{T} \\
\\
& \boxed{ \frac{\partial\cal L}{\partial h^{(L)}} = \frac{\partial\cal L}{\partial u^{(L)}} U^{T} }
\end{eqnarray}

### Parte 3c) Derivando desde las capas escondidas

\begin{equation}
\frac{\partial\cal L}{\partial u^{(k)}} = \frac{\partial\cal L}{\partial h^{(k)}} \frac{\partial\cal h^{(k)}}{\partial u^{(k)}} \\
\end{equation}

Para __sigmoid__ tenemos la siguiente derivada:
\begin{equation}
h^{(k)} = sig(u^{(k)})
\\
\frac{\partial\cal h^{(k)}}{\partial u^{(k)}} = h^{(k)}(1 - h^{(k)})
\end{equation}
<br><br>

Las siguientes derivadas pueden obtenerse independiente de la forma de la función de activación y son análogas a las calculadas en la última capa:
<br><br>
\begin{equation}
\frac{\partial\cal L}{\partial W^{(k)}} = (h^{(k)})^{T} \frac{\partial\cal L}{\partial u^{(k)}} \\
\end{equation}
<br><br>

\begin{equation}
\frac{\partial\cal L}{\partial b^{(k)}} = [1 \ldots 1] \frac{\partial\cal L}{\partial u^{(k)}} \\
\end{equation}
<br><br>

\begin{equation}
\frac{\partial\cal L}{\partial h^{(k-1)}} = \frac{\partial\cal L}{\partial u^{(k)}} (W^{(k)})^T \\
\end{equation}

### Otras derivadas (derivadas opcionales de celu y swish, de batch normalization, etc.)

### Parte 6a) Weight Decay

$w_{ij_{n+1}} = (1 - \frac{\lambda \alpha }{N})w_{ij_{n}} - \lambda \frac{\partial \cal L}{\partial w_{ij_{n}}}$
"""