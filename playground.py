import numpy as np
import activations as act
import optimizers as opti
from layers import Layer
from model import Model

x1 = np.random.uniform(-0.5, 0.5, (500, 5))
x2 = np.random.uniform(4.5, 5.5, (500, 5))

X = np.concatenate((x1, x2), axis = 0)
y = np.zeros((1000, 1))
y[:500] = 1

idx = list(range(X.shape[0]))
np.random.shuffle(idx)
X = X[idx]
y = y[idx]
#===================

x1t = np.random.uniform(-0.5, 0.5, (50, 5))
x2t = np.random.uniform(4.5, 5.5, (50, 5))

Xt = np.concatenate((x1t, x2t), axis=0)
yt = np.zeros((100, 1))
yt[:50] = 1
#===================

model = Model(X, y, int(3e1), 16)
optimizer = opti.ADAM(1e-1, 0.9, 0.99)
l1 = Layer(5, 3, act.Tanh(), optimizer)
l2 = Layer(3, 1, act.Sigmoid(), optimizer)

model.add_layer(l1)
model.add_layer(l2)

model.train()
model.plot_loss()
model.score(Xt, yt)