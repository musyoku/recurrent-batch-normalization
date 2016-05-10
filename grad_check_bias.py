import unittest

import numpy

import chainer
from chainer import cuda
from chainer import functions
from chainer.functions.connection import linear
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition
import bnlstm


class TestNonparameterizedLinear(unittest.TestCase):

	def setUp(self):
		self.b = numpy.random.uniform(1, 1, 3).astype(numpy.float32)

		self.x = numpy.random.uniform(-1, 1, (4, 3)).astype(numpy.float32)
		self.gy = numpy.random.uniform(-1, 1, (4, 3)).astype(numpy.float32)
		self.y = self.x + self.b

	def check_forward(self, x_data, b_data, y_expect):
		x = chainer.Variable(x_data)
		b = chainer.Variable(b_data)
		y = bnlstm.bias(x, b)
		gradient_check.assert_allclose(y_expect, y.data)

	@condition.retry(3)
	def test_forward_cpu(self):
		self.check_forward(self.x, self.b, self.x + self.b)

	@attr.gpu
	@condition.retry(3)
	def test_forward_gpu(self):
		self.check_forward(
			cuda.to_gpu(self.x), cuda.to_gpu(self.b), cuda.to_gpu(self.x + self.b))

	def check_backward(self, x_data, b_data, y_grad):
		args = (x_data, b_data)
		gradient_check.check_backward(bnlstm.BiasFunction(), args, y_grad, eps=1e-2)

	@condition.retry(3)
	def test_backward_cpu(self):
		self.check_backward(self.x, self.b, self.gy)

	@attr.gpu
	@condition.retry(3)
	def test_backward_gpu(self):
		self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.b), cuda.to_gpu(self.gy))

testing.run_module(__name__, __file__)