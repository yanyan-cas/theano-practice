#Scan Example: Computing tanh(x(t).dot(W) + b) elementwise
import theano
import theano.tensor as T
import numpy as np

#defining the tensor variables
X = T.matrix("X")
W = T.matrix("W")
b_sym = T.vector("b_sym")

results, updates = theano.scan(lambda v: T.tanh(T.dot(v, W) + b_sym), sequences=X)
compute_elementwise = theano.function(inputs=[X, W, b_sym], outputs=[results])

#test values
x = np.eye(2, dtype=theano.config.floatX)
w = np.ones((2, 2), dtype=theano.config.floatX)
b = np.ones((2), dtype=theano.config.floatX)

b[1] = 2

#print(compute_elementwise(x, w, b)[0])

#comparison with numpy
#print(np.tanh(x.dot(w) + b))

#Scan Example: Computing the sequence x(t) = tanh(x(t-1).dot(W)) + y(t).dot(U) + p(T-t).dot(V))

#import theano
#import theano.tensor as T
#import numpy as np

#define tensor variables
X = T.vector("X")
W = T.matrix("W")
b_sym = T.vector("b_sym")
U = T.matrix("U")
Y = T.matrix("Y")
V = T.matrix("V")
P = T.matrix("P")

results, updates = theano.scan(lambda y, p, x_tml: T.tanh(T.dot(x_tml, W) + T.dot(y, U) + T.dot(p, V)),
	sequences=[Y, P[::-1]], outputs_info=[X])

compute_seq = theano.function(inputs=[X, W, Y, U, P, V], outputs=[results])

#test values
x = np.zeros((2), dtype=theano.config.floatX)
x[1] = 1
w = np.ones((2,2), dtype=theano.config.floatX)
y = np.ones((5, 2), dtype=theano.config.floatX)
y[0, :] = -3
u = np.ones((2, 2), dtype=theano.config.floatX)
p = np.ones((5, 2), dtype=theano.config.floatX)
p[0, :] = 3
v = np.ones((2, 2), dtype=theano.config.floatX)

print(compute_seq(x, w, y, u, p, v)[0])

#comparison with numpy
x_res = np.zeros((5, 2), dtype=theano.config.floatX)
x_res[0] = np.tanh(x.dot(w) + y[0].dot(u) + p[4].dot(v))
for i in range(1, 5):
	x_res[i] = np.tanh(x_res[i - 1].dot(w) + y[i].dot(u) + p[4-i].dot(v))
print(x_res)








