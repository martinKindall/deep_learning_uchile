import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imresize

orig = np.loadtxt("escoba_train_mal2.txt")

# resize = imresize(orig, (128,128))

# relu = lambda x: 1 if x > 0 else 0
# vfunc = np.vectorize(relu)
# resize = vfunc(resize)

# np.savetxt("escoba_train.txt", resize, fmt="%d", delimiter=" ")

plt.imshow(orig, cmap='Greys')
plt.savefig('escoba_train_mal.png')