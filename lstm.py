import numpy as np
import theano
import theano.tensor as T

dtype=theano.config.floatX

# squashing of the gates should result in values between 0 and 1
# therefore we use the logistic function
sigma = lambda x: 1 / (1 + T.exp(-x))


# for the other activation function we use the tanh
act = T.tanh

# sequences: x_t
# prior results: h_tm1, c_tm1
# non-sequences: W_xi, W_hi, W_ci, b_i, W_xf, W_hf, W_cf, b_f, W_xc, W_hc, b_c, W_xy, W_hy, W_cy, b_y
def one_lstm_step(x_t, h_tm1, c_tm1, W_xi, W_hi, W_ci, b_i, W_xf, W_hf, W_cf, b_f, W_xc, W_hc, b_c, W_xy, W_ho, W_cy, b_o, W_hy, b_y):
    i_t = sigma(theano.dot(x_t, W_xi) + theano.dot(h_tm1, W_hi) + theano.dot(c_tm1, W_ci) + b_i)
    f_t = sigma(theano.dot(x_t, W_xf) + theano.dot(h_tm1, W_hf) + theano.dot(c_tm1, W_cf) + b_f)
    c_t = f_t * c_tm1 + i_t * act(theano.dot(x_t, W_xc) + theano.dot(h_tm1, W_hc) + b_c) 
    o_t = sigma(theano.dot(x_t, W_xo)+ theano.dot(h_tm1, W_ho) + theano.dot(c_t, W_co)  + b_o)
    h_t = o_t * act(c_t)
    y_t = sigma(theano.dot(h_t, W_hy) + b_y) 
    return [h_t, c_t, y_t]

#TODO: Use a more appropriate initialization method
def sample_weights(sizeX, sizeY):
    values = np.ndarray([sizeX, sizeY], dtype=dtype)
    for dx in xrange(sizeX):
        vals = np.random.uniform(low=-1., high=1.,  size=(sizeY,))
        #vals_norm = np.sqrt((vals**2).sum())
        #vals = vals / vals_norm
        values[dx,:] = vals
    _,svs,_ = np.linalg.svd(values)
    #svs[0] is the largest singular value                      
    values = values / svs[0]
    print sizeX
    print sizeY
    return values  

print 'Setting up nn'

n_in = 7 # for embedded reber grammar
n_hidden = n_i = n_c = n_o = n_f = 10
n_y = 7 # for embedded reber grammar

# initialize weights
# i_t and o_t should be "open" or "closed"
# f_t should be "open" (don't forget at the beginning of training)
# we try to archive this by appropriate initialization of the corresponding biases 

print 'Setting up weights'

W_xi = theano.shared(sample_weights(n_in, n_i))  
print 'W_xi' + str(W_xi.shape.eval())

W_hi = theano.shared(sample_weights(n_hidden, n_i))  
print 'W_hi' + str(W_hi.shape.eval())

W_ci = theano.shared(sample_weights(n_c, n_i))  
print 'W_ci' + str(W_ci.shape.eval())

b_i = theano.shared(np.cast[dtype](np.random.uniform(-0.5,.5,size = n_i)))
print 'b_i' + str(b_i.shape.eval())

W_xf = theano.shared(sample_weights(n_in, n_f)) 
print 'W_xf' + str(W_xf.shape.eval())

W_hf = theano.shared(sample_weights(n_hidden, n_f))
print 'W_hi' + str(W_hi.shape.eval())

W_cf = theano.shared(sample_weights(n_c, n_f))
print 'W_cf' + str(W_cf.shape.eval())

b_f = theano.shared(np.cast[dtype](np.random.uniform(0, 1.,size = n_f)))
print 'b_f' + str(b_f.shape.eval())

W_xc = theano.shared(sample_weights(n_in, n_c))  
print 'W_xc' + str(W_xc.shape.eval())

W_hc = theano.shared(sample_weights(n_hidden, n_c))
print 'W_hc' + str(W_hc.shape.eval())

b_c = theano.shared(np.zeros(n_c, dtype=dtype))
print 'b_c' + str(b_c.shape.eval())

W_xo = theano.shared(sample_weights(n_in, n_o))
print 'W_xo' + str(W_xo.shape.eval())

W_ho = theano.shared(sample_weights(n_hidden, n_o))
print 'W_ho' + str(W_ho.shape.eval())

W_co = theano.shared(sample_weights(n_c, n_o))
print 'W_co' + str(W_co.shape.eval())

b_o = theano.shared(np.cast[dtype](np.random.uniform(-0.5,.5,size = n_o)))
print 'b_o' + str(b_o.shape.eval())

W_hy = theano.shared(sample_weights(n_hidden, n_y))
print 'W_hy' + str(W_hy.shape.eval())

b_y = theano.shared(np.zeros(n_y, dtype=dtype))
print 'b_y' + str(b_y.shape.eval())

c0 = theano.shared(np.zeros(n_hidden, dtype=dtype))
print 'c0' + str(c0.shape.eval())

h0 = T.tanh(c0)
print 'h0' + str(h0)

params = [W_xi, W_hi, W_ci, b_i, W_xf, W_hf, W_cf, b_f, W_xc, W_hc, b_c, W_xo, W_ho, W_co, b_o, W_hy, b_y, c0]

#first dimension is time

#input 
v = T.matrix(dtype=dtype)

# target
target = T.matrix(dtype=dtype)

# hidden and outputs of the entire sequence
[h_vals, _, y_vals], _ = theano.scan(fn=one_lstm_step, 
                                  sequences = dict(input=v, taps=[0]), 
                                  outputs_info = [h0, c0, None ], # corresponds to return type of fn
                                  non_sequences = [W_xi, W_hi, W_ci, b_i, W_xf, W_hf, W_cf, b_f, W_xc, W_hc, b_c, W_xo, W_ho, W_co, b_o, W_hy, b_y] )

cost = -T.mean(target * T.log(y_vals)+ (1.- target) * T.log(1. - y_vals))

# learning rate
lr = np.cast[dtype](.1)
learning_rate = theano.shared(lr)

gparams = []
for param in params:
  gparam = T.grad(cost, param)
  gparams.append(gparam)

updates=[]
for param, gparam in zip(params, gparams):
    updates.append((param, param - gparam * learning_rate))

print 'Setting up training data'
import reberGrammar
train_data = reberGrammar.get_n_embedded_examples(1000)

print 'Setting up learning rnn'
learn_rnn_fn = theano.function(inputs = [v, target],
                               outputs = cost,
                               updates = updates)

print 'Setting up training'
nb_epochs=250
train_errors = np.ndarray(nb_epochs)
def train_rnn(train_data):      
  for x in range(nb_epochs):
    error = 0.
    print x
    for j in range(len(train_data)):  
        index = np.random.randint(0, len(train_data))
        i, o = train_data[index]
        train_cost = learn_rnn_fn(i, o)
        error += train_cost
    train_errors[x] = error 
    
print 'Training'
train_rnn(train_data)

import matplotlib.pyplot as plt
plt.plot(np.arange(nb_epochs), train_errors, 'b-')
plt.xlabel('epochs')
plt.ylabel('error')
plt.ylim(0., 50)
plt.show()

predictions = theano.function(inputs = [v], outputs = y_vals)

test_data = reberGrammar.get_n_embedded_examples(10)

def print_out(test_data):
    for i,o in test_data:
        p = predictions(i)
        print o[-2] # target
        print p[-2] # prediction
        print 
print_out(test_data)
