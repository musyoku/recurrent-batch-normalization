import unittest
import numpy as np
import chainer
from chainer import cuda, functions, gradient_check, links, testing
from chainer.testing import attr
import bnlstm

@testing.parameterize(
	{"in_size": 10, "out_size": 10},
	{"in_size": 10, "out_size": 40},
)
class TestLSTM(unittest.TestCase):

	def setUp(self):
		self.link = bnlstm.BNLSTM(self.in_size, self.out_size)
		wx = self.link.wx.W.data
		wx[...] = np.random.uniform(-1, 1, wx.shape)
		wh = self.link.wh.W.data
		wh[...] = np.random.uniform(-1, 1, wh.shape)
		self.link.zerograds()

		self.wx = wx.copy()  # fixed on CPU
		self.wh = wh.copy()  # fixed on CPU

		x_shape = (4, self.in_size)
		self.x = np.random.uniform(-1, 1, x_shape).astype(np.float32)

	def check_forward(self, x_data):
		xp = self.link.xp
		x = chainer.Variable(x_data)
		h1 = self.link(x)
		c0 = chainer.Variable(xp.zeros((len(self.x), self.out_size), dtype=self.x.dtype))
		lstm_in = self.link.bnx(self.link.wx(x))
		c1_expect = bnlstm.bn_lstm_cell(c0, lstm_in)
		h1_expect = bnlstm.bn_lstm_state(self.link.bnc(c1_expect), lstm_in)
		gradient_check.assert_allclose(h1.data, h1_expect.data)
		gradient_check.assert_allclose(self.link.h.data, h1_expect.data)
		gradient_check.assert_allclose(self.link.c.data, c1_expect.data)

		h2 = self.link(x)
		lstm_in = self.link.bnx(self.link.wx(x)) + self.link.bnh(self.link.wh(h1))
		c2_expect = bnlstm.bn_lstm_cell(c1_expect, lstm_in)
		h2_expect = bnlstm.bn_lstm_state(self.link.bnc(c2_expect), lstm_in)
		gradient_check.assert_allclose(h2.data, h2_expect.data)

	def test_forward_cpu(self):
		self.check_forward(self.x)

	@attr.gpu
	def test_forward_gpu(self):
		self.link.to_gpu()
		self.check_forward(cuda.to_gpu(self.x))

	def check_backward(self, x_data):
		xp = self.link.xp
		x = chainer.Variable(x_data)
		y_shape = (4, self.out_size)
		y_grad = np.random.uniform(-1, 1, y_shape).astype(np.float32)
		if xp is cuda.cupy:
			y_grad = cuda.to_gpu(y_grad)

		c0 = chainer.Variable(xp.zeros((len(self.x), self.out_size), dtype=self.x.dtype))
		lstm_in = self.link.bnx(self.link.wx(x))
		c1 = bnlstm.bn_lstm_cell(c0, lstm_in)
		h1 = bnlstm.bn_lstm_state(self.link.bnc(c1), lstm_in)
		gradient_check.check_backward(bnlstm.CellState(), (c0.data, lstm_in.data), y_grad, eps=1e-2)
		gradient_check.check_backward(bnlstm.HiddenState(), (self.link.bnc(c1).data, lstm_in.data), y_grad, eps=1e-2)

		lstm_in = self.link.bnx(self.link.wx(x)) + self.link.bnh(self.link.wh(h1))
		c2 = bnlstm.bn_lstm_cell(c1, lstm_in)
		h2 = bnlstm.bn_lstm_state(self.link.bnc(c2), lstm_in)
		gradient_check.check_backward(bnlstm.CellState(), (c2.data, lstm_in.data), y_grad, eps=1e-2)
		gradient_check.check_backward(bnlstm.HiddenState(), (self.link.bnc(c2).data, lstm_in.data), y_grad, eps=1e-2)

	def test_backward_cpu(self):
		self.check_backward(self.x)

	@attr.gpu
	def test_backward_gpu(self):
		self.link.to_gpu()
		self.check_backward(cuda.to_gpu(self.x))

class TestLSSTMRestState(unittest.TestCase):

	def setUp(self):
		self.link = bnlstm.BNLSTM(5, 7)
		self.x = chainer.Variable(
			np.random.uniform(-1, 1, (3, 5)).astype(np.float32))

	def check_state(self):
		self.assertIsNone(self.link.c)
		self.assertIsNone(self.link.h)
		self.link(self.x)
		self.assertIsNotNone(self.link.c)
		self.assertIsNotNone(self.link.h)

	def test_state_cpu(self):
		self.check_state()

	@attr.gpu
	def test_state_gpu(self):
		self.link.to_gpu()
		self.x.to_gpu()
		self.check_state()

	def check_reset_state(self):
		self.link(self.x)
		self.link.reset_state()
		self.assertIsNone(self.link.c)
		self.assertIsNone(self.link.h)

	def test_reset_state_cpu(self):
		self.check_reset_state()

	@attr.gpu
	def test_reset_state_gpu(self):
		self.link.to_gpu()
		self.x.to_gpu()
		self.check_reset_state()


class TestLSTMToCPUToGPU(unittest.TestCase):

	def setUp(self):
		self.link = bnlstm.BNLSTM(5, 7)
		self.x = chainer.Variable(
			np.random.uniform(-1, 1, (3, 5)).astype(np.float32))

	def check_to_cpu(self, s):
		self.link.to_cpu()
		self.assertIsInstance(s.data, self.link.xp.ndarray)
		self.link.to_cpu()
		self.assertIsInstance(s.data, self.link.xp.ndarray)

	def test_to_cpu_cpu(self):
		self.link(self.x)
		self.check_to_cpu(self.link.c)
		self.check_to_cpu(self.link.h)

	@attr.gpu
	def test_to_cpu_gpu(self):
		self.link.to_gpu()
		self.x.to_gpu()
		self.link(self.x)
		self.check_to_cpu(self.link.c)
		self.check_to_cpu(self.link.h)

	def check_to_cpu_to_gpu(self, s):
		self.link.to_gpu()
		self.assertIsInstance(s.data, self.link.xp.ndarray)
		self.link.to_gpu()
		self.assertIsInstance(s.data, self.link.xp.ndarray)
		self.link.to_cpu()
		self.assertIsInstance(s.data, self.link.xp.ndarray)
		self.link.to_gpu()
		self.assertIsInstance(s.data, self.link.xp.ndarray)

	@attr.gpu
	def test_to_cpu_to_gpu_cpu(self):
		self.link(self.x)
		self.check_to_cpu_to_gpu(self.link.c)
		self.check_to_cpu_to_gpu(self.link.h)

	@attr.gpu
	def test_to_cpu_to_gpu_gpu(self):
		self.link.to_gpu()
		self.x.to_gpu()
		self.link(self.x)
		self.check_to_cpu_to_gpu(self.link.c)
		self.check_to_cpu_to_gpu(self.link.h)

testing.run_module(__name__, __file__)