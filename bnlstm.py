# -*- coding: utf-8 -*-
import numpy as np
from chainer import cuda, Variable, optimizers, function, link
from chainer import functions as F
from chainer import links as L
from chainer.utils import type_check

class NoBetaBatchNormalizationFunction(function.Function):

	def __init__(self, eps=1e-5):
		self.eps = eps

	def check_type_forward(self, in_types):
		n_in = in_types.size().eval()
		if n_in != 2 and n_in != 4:
			raise type_check.InvalidType(
				"%s or %s" % (in_types.size() == 2, in_types.size() == 4),
				"%s == %s" % (in_types.size(), n_in))

		x_type, gamma_type = in_types[:2]
		type_check.expect(
			x_type.dtype == np.float32,
			x_type.ndim >= gamma_type.ndim + 1,
			gamma_type.dtype == np.float32,
		)

		if len(in_types) == 4:
			mean_type, var_type = in_types[2:]
			type_check.expect(
				mean_type.dtype == np.float32,
				mean_type.shape == gamma_type.shape,
				var_type.dtype == np.float32,
				var_type.shape == gamma_type.shape,
			)

	def forward(self, inputs):
		xp = cuda.get_array_module(*inputs)
		x, gamma = inputs[:2]

		head_ndim = gamma.ndim + 1
		expander = (None, Ellipsis) + (None,) * (x.ndim - head_ndim)
		gamma = gamma[expander]

		if len(inputs) == 4:
			mean = inputs[2]
			var = inputs[3]
		else:
			axis = (0,) + tuple(range(head_ndim, x.ndim))
			mean = x.mean(axis=axis)
			var = x.var(axis=axis)
			var += self.eps
			self.mean = mean
			self.var = var

		self.std = xp.sqrt(var, dtype=var.dtype)

		if xp is np:
			x_mu = x - mean[expander]
			self.x_hat = x_mu / self.std[expander]
			y = gamma * self.x_hat
		else:
			self.x_hat, y = cuda.elementwise(
				"T x, T mean, T std, T gamma", "T x_hat, T y",
				"""
				   x_hat = (x - mean) / std;
				   y = gamma * x_hat;
				""",
				"bn_fwd")(x, mean[expander], self.std[expander], gamma)
		return y,

	def backward(self, inputs, grad_outputs):
		x, gamma = inputs[:2]
		gy = grad_outputs[0]

		head_ndim = gamma.ndim + 1
		expander = (None, Ellipsis) + (None,) * (x.ndim - head_ndim)
		m = gamma.dtype.type(x.size // gamma.size)

		xp = cuda.get_array_module(x)

		axis = (0,) + tuple(range(head_ndim, x.ndim))
		gbeta = gy.sum(axis=axis)
		ggamma = (gy * self.x_hat).sum(axis=axis)

		if len(inputs) == 4:
			var = inputs[3]
			gs = gamma / self.std
			gmean = -gs * gbeta
			gvar = -0.5 * gamma / var * ggamma
			gx = gs[expander] * gy
			gbeta = xp.full_like(gbeta, 0.0)
			return gx, ggamma, gmean, gvar

		if xp is np:
			gx = (gamma / self.std)[expander] * (
				gy - (self.x_hat * ggamma[expander] + gbeta[expander]) / m)
		else:
			inv_m = np.float32(1) / m
			gx = cuda.elementwise(
				"T gy, T x_hat, T gamma, T std, T ggamma, T gbeta, T inv_m",
				"T gx",
				"gx = (gamma / std) * (gy - (x_hat * ggamma + gbeta) * inv_m)",
				"bn_bwd")(gy, self.x_hat, gamma[expander], self.std[expander],
						  ggamma[expander], gbeta[expander], inv_m)
		return gx, ggamma

def batch_normalization(x, gamma, eps=1e-5):
	return NoBetaBatchNormalizationFunction(eps)(x, gamma)

def fixed_batch_normalization(x, gamma, mean, var, eps=1e-5):
	return NoBetaBatchNormalizationFunction(eps)(x, gamma, mean, var)

class NoBetaBatchNormalization(link.Link):

	def __init__(self, size, initial_gamma=0.1, decay=0.9, eps=1e-5, dtype=np.float32):
		super(NoBetaBatchNormalization, self).__init__()
		self.add_param("gamma", size, dtype=dtype)
		self.gamma.data.fill(initial_gamma)
		self.add_persistent("avg_mean", np.zeros(size, dtype=dtype))
		self.add_persistent("avg_var", np.zeros(size, dtype=dtype))
		self.add_persistent("N", 0)
		self.decay = decay
		self.eps = eps

	def __call__(self, x, test=False, finetune=False):
		use_batch_mean = not test or finetune

		if use_batch_mean:
			func = NoBetaBatchNormalizationFunction(self.eps)
			ret = func(x, self.gamma)

			if finetune:
				self.N += 1
				decay = 1. - 1. / self.N
			else:
				decay = self.decay

			with cuda.get_device(x.data):
				m = x.data.size // self.gamma.data.size
				adjust = m / max(m - 1., 1.)  # unbiased estimation
				self.avg_mean *= decay
				func.mean *= 1 - decay  # reuse buffer as a temporary
				self.avg_mean += func.mean
				del func.mean
				self.avg_var *= decay
				func.var *= (1 - decay) * adjust  # reuse buffer as a temporary
				self.avg_var += func.var
				del func.var
		else:
			mean = Variable(self.avg_mean, volatile="auto")
			var = Variable(self.avg_var, volatile="auto")
			ret = fixed_batch_normalization(x, self.gamma, mean, var, self.eps)
		return ret

	def start_finetuning(self):
		self.N = 0


class BatchNormalization(L.BatchNormalization):

	def __init__(self, size, initial_gamma=0.1, decay=0.9, eps=1e-5, dtype=np.float32):
		super(L.BatchNormalization, self).__init__()
		self.add_param('gamma', size, dtype=dtype)
		self.gamma.data.fill(initial_gamma)
		self.add_param('beta', size, dtype=dtype)
		self.beta.data.fill(0)
		self.add_persistent('avg_mean', np.zeros(size, dtype=dtype))
		self.add_persistent('avg_var', np.zeros(size, dtype=dtype))
		self.add_persistent('N', 0)
		self.decay = decay
		self.eps = eps
		
def _extract_gates(x):
	r = x.reshape((x.shape[0], x.shape[1] // 4, 4) + x.shape[2:])
	return (r[:, :, i] for i in range(4))

def _sigmoid(x):
	xp = cuda.get_array_module(*(x,))
	return 1 / (1 + xp.exp(-x))

def _grad_sigmoid(x):
	return x * (1 - x)


def _grad_tanh(x):
	return 1 - x * x

class CellState(function.Function):
	def check_type_forward(self, in_types):
		type_check.expect(in_types.size() == 2)
		c_type, x_type = in_types

		type_check.expect(
			c_type.dtype == np.float32,
			x_type.dtype == np.float32,
			c_type.ndim >= 2,
			x_type.ndim >= 2,
			c_type.ndim == x_type.ndim,
			x_type.shape[0] == c_type.shape[0],
			x_type.shape[1] == 4 * c_type.shape[1],
		)
		for i in range(2, c_type.ndim.eval()):
			type_check.expect(x_type.shape[i] == c_type.shape[i])

	def forward(self, inputs):
		xp = cuda.get_array_module(*inputs)
		c_prev, x = inputs
		g, i, f, o = _extract_gates(x)
		self.g = xp.tanh(g)
		self.i = _sigmoid(i)
		self.f = _sigmoid(f)
		self.c = self.g * self.i + self.f * c_prev
		return self.c,

	def backward(self, inputs, grad_outputs):
		xp = cuda.get_array_module(*inputs)
		c_prev, x = inputs
		gc, = grad_outputs

		gx = xp.empty_like(x)
		gg, gi, gf, go = _extract_gates(gx)

		if gc is None:
			gc = 0

		co = xp.tanh(self.c)
		gg[:] = gc * self.i * _grad_tanh(self.g)
		gi[:] = gc * self.g * _grad_sigmoid(self.i)
		gf[:] = gc * c_prev * _grad_sigmoid(self.f)
		go[:] = 0
		gc_prev = gc * self.f

		return gc_prev, gx


class HiddenState(CellState):

	def forward(self, inputs):
		xp = cuda.get_array_module(*inputs)
		c, x = inputs
		g, i, f, o = _extract_gates(x)
		self.o = _sigmoid(o)
		self.c = xp.tanh(c)
		self.h = self.o * self.c
		return self.h,

	def backward(self, inputs, grad_outputs):
		xp = cuda.get_array_module(*inputs)
		c, x = inputs
		gh, = grad_outputs

		gx = xp.empty_like(x)
		gg, gi, gf, go = _extract_gates(gx)

		if gh is None:
			gh = 0

		gg[:] = 0
		gi[:] = 0
		gf[:] = 0
		go[:] = gh * self.c * _grad_sigmoid(self.o)
		gc = gh * _grad_tanh(self.c) * self.o

		return gc, gx

def bn_lstm_cell(c_prev, x):
	return CellState()(c_prev, x)

def bn_lstm_state(c, x):
	return HiddenState()(c, x)
	


def _as_mat(x):
	if x.ndim == 2:
		return x
	return x.reshape(len(x), -1)


class BiasFunction(function.Function):

	def check_type_forward(self, in_types):
		n_in = in_types.size()
		type_check.expect(2 <= n_in, n_in <= 3)
		x_type = in_types[0]

		type_check.expect(
			x_type.dtype == np.float32,
			x_type.ndim >= 2,
		)
		if n_in.eval() == 3:
			b_type = in_types[1]
			type_check.expect(
				b_type.dtype == np.float32,
				b_type.ndim == 1,
			)

	def forward(self, inputs):
		x = _as_mat(inputs[0])
		b = inputs[1]
		return x + b,

	def backward(self, inputs, grad_outputs):
		x = _as_mat(inputs[0])
		gy = grad_outputs[0]
		gb = gy.sum(0)
		return gy, gb

def bias(x, b):
	return BiasFunction()(x, b)

class Bias(link.Link):

	def __init__(self, in_size, out_size, bias=0, nobias=False, initialW=None, initial_bias=None):
		super(Bias, self).__init__()
		self.add_param('b', out_size)
		if initial_bias is None:
			initial_bias = bias
		self.b.data[...] = initial_bias

	def __call__(self, x):
		return bias(x, self.b)

class BNLSTM(link.Chain):

	def __init__(self, in_size, out_size):
		super(BNLSTM, self).__init__(
			wx=L.Linear(in_size, 4 * out_size, nobias=True),
			wh=L.Linear(out_size, 4 * out_size, nobias=True),
			bias=Bias(4 * out_size, 4 * out_size),
			bnx=NoBetaBatchNormalization(4 * out_size),
			bnh=NoBetaBatchNormalization(4 * out_size),
			bnc=BatchNormalization(out_size)
		)
		self.state_size = out_size
		self.reset_state()

	def to_cpu(self):
		super(BNLSTM, self).to_cpu()
		if self.c is not None:
			self.c.to_cpu()
		if self.h is not None:
			self.h.to_cpu()

	def to_gpu(self, device=None):
		super(BNLSTM, self).to_gpu(device)
		if self.c is not None:
			self.c.to_gpu(device)
		if self.h is not None:
			self.h.to_gpu(device)

	def reset_state(self):
		self.c = self.h = None

	def __call__(self, x, test=False):
		lstm_in = self.bnx(self.wx(x), test=test)
		if self.h is not None:
			lstm_in += self.bnh(self.wh(self.h), test=test)
		lstm_in = self.bias(lstm_in)
		if self.c is None:
			xp = self.xp
			self.c = Variable(xp.zeros((len(x.data), self.state_size), dtype=x.data.dtype), volatile="auto")
		self.c = bn_lstm_cell(self.c, lstm_in)
		self.h = bn_lstm_state(self.bnc(self.c, test=test), lstm_in)
		return self.h