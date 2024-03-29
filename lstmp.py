import theano
import theano.tensor as T
import numpy as np

# LSTM network
class LSTMP(object):
		
	def __init__(self, n_in, n_hidden, n_out):
		dtype = theano.config.floatX
		n_i = n_f = n_c = n_o = n_hidden
		n_p = n_hidden
		n_r = 256
		#Init weights	
		def init_weights(start, end):
			values = np.random.uniform(low=-0.1, high=0.1, size=(start, end))
			return values
		#Init params		
		#c0 = theano.shared(np.random.uniform(low=-0.1, high=0.1, size=n_c))
		#r0 = theano.shared(np.random.uniform(low=-0.1, high=0.1, size=n_r))
		#Input Gate params
		W_xi = theano.shared(init_weights(n_in, n_i))
		W_hi = theano.shared(init_weights(n_r, n_i))
		W_ci = theano.shared(init_weights(n_c, n_i))
		b_i = theano.shared(np.random.uniform(low=-0.1, high=0.1, size=n_i))
		#Forget Gate params
		W_xf = theano.shared(init_weights(n_in, n_f))
		W_hf = theano.shared(init_weights(n_r, n_f))
		W_cf = theano.shared(init_weights(n_c, n_f))
		b_f = theano.shared(np.random.uniform(low=0., high=1., size=n_f))
		#Cell params
		W_xc = theano.shared(init_weights(n_in, n_c))
		W_hc = theano.shared(init_weights(n_r, n_c))
		b_c = theano.shared(np.zeros(n_c, dtype=dtype))
		#Output Gate params
		W_xo = theano.shared(init_weights(n_in, n_o))
		W_ho = theano.shared(init_weights(n_r, n_o))
		W_co = theano.shared(init_weights(n_c, n_o))
		b_o = theano.shared(np.random.uniform(low=-0.1, high=0.1, size=n_o))
		#Recurrent Projection params
		W_hr = theano.shared(init_weights(n_hidden, n_r))
		W_hp = theano.shared(init_weights(n_hidden, n_p))
		#Output params	
		W_ry = theano.shared(init_weights(n_r, n_out))
		W_py = theano.shared(init_weights(n_p, n_out))
		b_y = theano.shared(np.zeros(n_out, dtype=dtype))
		#Params
		params = [W_xi, W_hi, W_ci, b_i, W_xf, W_hf, W_cf, b_f,
				  W_xc, W_hc, b_c, W_xo, W_ho, W_co, b_o,
				  W_hr, W_hp, W_ry, W_py, b_y]
		self.params = params
				
		#Tensor variables
		r0 = T.vector()
		c0 = T.vector()
		x = T.matrix()
		t = T.matrix()
		lr = T.scalar()
		#LSTM http://arxiv.org/abs/1402.1128
		[r, c, z], _ = theano.scan(fn = self.recurrent_fn, sequences = x,
                             outputs_info  = [r0, c0, None], #corresponds to return type of fn
                             non_sequences = [W_xi, W_hi, W_ci, b_i, W_xf, W_hf, W_cf, b_f,
				                              W_xc, W_hc, b_c, W_xo, W_ho, W_co, b_o,
				                              W_hr, W_hp, W_ry, W_py, b_y])

        #Cost
		cost = (T.sqr(t - z)).mean()
		# Negative Log-Likelihood
		# LL = $\prod 1/Z * exp( -(f(x)-t)^2 / (2*sigma^2) )
		# Z = sqrt(2pi * sqrt(2*sigma^2))
		nll = (T.sqr(t - z)).sum()/(2*cost) + T.log(T.sqrt(2*np.pi*cost))
		
		#Updates
		updates = self.RMSprop(cost, params, learnrate=lr)

		#Theano Functions
		self.train = theano.function([r0, c0, x, t, lr], [r[-1], c[-1], z, cost], 
                                     on_unused_input='warn', 
                                     updates=updates)						 
		self.predict = theano.function([r0, c0, x], [r[-1], c[-1], z])	
		
    #LSTM step
	def recurrent_fn(self, x_t, r_tm1, c_tm1,
                         W_xi, W_hi, W_ci, b_i, W_xf, W_hf, W_cf, b_f,
				         W_xc, W_hc, b_c, W_xo, W_ho, W_co, b_o,
				         W_hr, W_hp, W_ry, W_py, b_y):
		#Input Gate
		i_t = T.nnet.sigmoid(T.dot(x_t, W_xi) + T.dot(r_tm1, W_hi) + T.dot(c_tm1, W_ci) + b_i)
		#Forget Gate
		f_t = T.nnet.sigmoid(T.dot(x_t, W_xf) + T.dot(r_tm1, W_hf) + T.dot(c_tm1, W_cf) + b_f)		
		#Cell
		c_t = f_t*c_tm1 + i_t*T.tanh(T.dot(x_t, W_xc) + T.dot(r_tm1, W_hc) + b_c)
		#Output Gate
		o_t = T.nnet.sigmoid(T.dot(x_t, W_xo) + T.dot(r_tm1, W_ho) + T.dot(c_t, W_co) + b_o)
		#LSTMemory Block Output
		h_t = o_t * T.tanh(c_t)
		#Recurrent Projection layer
		r_t = T.dot(h_t, W_hr)
		p_t = T.dot(h_t, W_hp)
		#Output
		y_t = T.dot(r_t, W_ry) + T.dot(p_t, W_py) + b_y
		return [r_t, c_t, y_t]
	
	#RMSprop
	def RMSprop(self, cost, params, learnrate, rho=0.90, epsilon=1e-6):
		gparams = []
		for param in params:
			gparam = T.grad(cost, param)
			gparams.append(gparam)	
		updates=[]
		for param, gparam in zip(params, gparams):
			acc = theano.shared(param.get_value() * 0.)
			acc_new = rho * acc + (1 - rho) * gparam ** 2
			gradient_scaling = T.sqrt(acc_new + epsilon)
			gparam = gparam / gradient_scaling
			updates.append((acc, acc_new))
			updates.append((param, param - gparam * learnrate))
		return updates
