# computing trace of X
import theano
import theano.tensor as T
import numpy as np

floatX = "float32"

#define tensor variable
X = T.matrix("X")
results, updates = theano.scan(lambda i, j, t_f: T.cast(X[i, j] + t_f, floatX),
	sequences=[T.arange(X.shape[0]), T.arange(X.shape[1])], outputs_info=np.asarray(0., dtype=floatX))

result = results[-1]
compute_trace = theano.function(inputs=[X], outputs=[result])

#test value
x = np.eye(5, dtype=theano.config.floatX)
x[0] = np.arange(5, dtype=theano.config.floatX)

print(compute_trace(x)[0])

print(np.diagonal(x).sum())

