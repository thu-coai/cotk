# coding:utf-8
import torch
import torch.nn.functional as F
from torch.autograd import Variable, Function

from .cuda_helper import TensorType

def gumbel_max(inp, alpha, beta):
	assert len(inp.size()) == 2
	g = TensorType()(inp.size()).uniform_(0.0001, 0.9999)
	g = Variable(-torch.log(-torch.log(g)))
	inp_g = F.softmax((F.log_softmax(inp, dim=1) + g * alpha) * beta, dim=1)
	return StraightThrough.apply(inp_g)

def gumbel_max_binary(inp, alpha, beta):
	inp = torch.cat([1-inp, inp], 1)
	g = TensorType()(inp.size()).uniform_(0.0001, 0.9999)
	g = Variable(-torch.log(-torch.log(g)))
	inp_g = F.softmax((torch.log(inp) + g * alpha) * beta, dim=1)
	return StraightThrough.apply(inp_g)

# pylint: disable=W0221
class StraightThrough(Function):
	@staticmethod
	def forward(ctx, inp):
		#ctx.save_for_backward(inp)
		_, idx = torch.max(inp, dim=1)
		output = TensorType()(*inp.size()).zero_()
		output[range(inp.size()[0]), idx] = 1
		return output, idx

	@staticmethod
	def backward(ctx, grad_output, _):
		#inp = ctx.saved_variables
		return grad_output.clone()
    