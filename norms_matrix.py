# Scan example: computing norms of lines of X
import theano
import theano.tensor as T
import numpy as np

#define tensor variable
X = T.matrix("X")
results, updates = theano.scan(lambda x_i: T.sqrt((x_i ** 2).sum()), sequences=[X])
compute_norm_lines = theano.function(inputs=[X], outputs=[results])

#test value
x = np.diag(np.arange(1, 6, dtype=theano.config.floatX), 1)
print(compute_norm_lines(x)[0])

#comparison with numpy
print(np.sqrt((x ** 2).sum(1)))

#computing norms of columns of X

#define tensor variable
X = T.matrix("X")
results, updates = theano.scan(lambda x_i: T.sqrt((x_i ** 2).sum()), sequences=[X.T])
compute_norm_cols = theano.function(inputs=[X], outputs=[results])

#test value
x= np.diag(np.arange(1, 6, dtype=theano.config.floatX), 1)
print(compute_norm_cols(x)[0])

#comparison with numpy
print(np.sqrt((x ** 2).sum(0)))

