# -*- coding: utf-8 -*-
import numpy as np
from chainer import cuda, Variable, optimizers, function, link
from chainer import functions as F
from chainer import links as L
from chainer.utils import type_check

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
	
class BNLSTM(link.Chain):

	def __init__(self, in_size, out_size):
		super(BNLSTM, self).__init__(
			wx=L.Linear(in_size, 4 * out_size),
			wh=L.Linear(out_size, 4 * out_size, nobias=True),
			bnx=L.BatchNormalization(4 * out_size),
			bnh=L.BatchNormalization(4 * out_size),
			bnc=L.BatchNormalization(out_size)
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
		lstm_in = self.bnx(self.wx(x))
		if self.h is not None:
			lstm_in += self.bnh(self.wh(self.h))
		if self.c is None:
			xp = self.xp
			self.c = Variable(xp.zeros((len(x.data), self.state_size), dtype=x.data.dtype), volatile="auto")
		self.c = bn_lstm_cell(self.c, lstm_in)
		self.h = bn_lstm_state(self.bnc(self.c), lstm_in)
		return self.h