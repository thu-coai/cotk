# coding:utf-8
import torch
import torch.nn.functional as F
from torch.autograd import Variable, Function

from .cuda_helper import Tensor, cuda

def gumbel_softmax(logits, tau=1, dim=-1):
	# type: (Tensor, float, bool, float, int) -> Tensor
	r"""
	Samples from the Gumbel-Softmax distribution (`Link 1`_  `Link 2`_) and optionally discretizes.

	Args:
	  logits: `[..., num_features]` unnormalized log probabilities
	  tau: non-negative scalar temperature
	  hard: if ``True``, the returned samples will be discretized as one-hot vectors,
			but will be differentiated as if it is the soft sample in autograd
	  dim (int): A dimension along which softmax will be computed. Default: -1.

	Returns:
	  Sampled tensor of same shape as `logits` from the Gumbel-Softmax distribution.
	  If ``hard=True``, the returned samples will be one-hot, otherwise they will
	  be probability distributions that sum to 1 across `dim`.
	"""

	gumbels = -torch.empty_like(logits).exponential_().log()  # ~Gumbel(0,1)
	gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
	y_soft = gumbels.softmax(dim)

	ret = y_soft
	return ret

def gumbel_max(logits, tau=1, dim=-1):
	gumbels = -torch.empty_like(logits).exponential_().log()  # ~Gumbel(0,1)
	gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
	y_soft = gumbels.softmax(dim)

	# Straight through.
	index = y_soft.max(dim, keepdim=True)[1]
	y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
	ret = y_hard - y_soft.detach() + y_soft
	return ret

def gumbel_max_with_mask(logits, mask, tau=1, dim=-1):
	gumbels = -torch.empty_like(logits).exponential_().log()  # ~Gumbel(0,1)
	gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
	gumbels = gumbels.masked_fill(mask==0, -1e9)
	y_soft = gumbels.softmax(dim)

	# Straight through.
	index = y_soft.max(dim, keepdim=True)[1]
	y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
	ret = y_hard - y_soft.detach() + y_soft
	return ret

# def gumbel_softmax(inp, alpha, beta):
# 	g = Tensor(inp.size()).uniform_(0.0001, 0.9999)
# 	g = Variable(-torch.log(-torch.log(g)))
# 	inp_g = F.softmax((F.log_softmax(inp, dim=-1) + g * alpha) * beta, dim=-1)
# 	return inp_g

# def gumbel_max(inp, alpha, beta):
# 	g = Tensor(inp.size()).uniform_(0.0001, 0.9999)
# 	g = Variable(-torch.log(-torch.log(g)))
# 	inp_g = F.softmax((F.log_softmax(inp, dim=1) + g * alpha) * beta, dim=1)
# 	shape = inp_g.shape
# 	output, idx = StraightThrough.apply(inp_g.reshape(-1, shape[-1]))
# 	return output.reshape(*shape), idx.reshape(*(list(shape[:-1]) + [-1]))

# def gumbel_max_binary(inp, alpha, beta):
# 	inp = torch.cat([1-inp, inp], 1)
# 	g = Tensor(inp.size()).uniform_(0.0001, 0.9999)
# 	g = Variable(-torch.log(-torch.log(g)))
# 	inp_g = F.softmax((torch.log(inp) + g * alpha) * beta, dim=1)
# 	return StraightThrough.apply(inp_g)

# # pylint: disable=W0221
# class StraightThrough(Function):
# 	@staticmethod
# 	def forward(ctx, inp):
# 		#ctx.save_for_backward(inp)
# 		_, idx = torch.max(inp, dim=1)
# 		output = Tensor(inp.size()).zero_()
# 		output[range(inp.size()[0]), idx] = 1
# 		return output, idx

# 	@staticmethod
# 	def backward(ctx, grad_output, _):
# 		#inp = ctx.saved_variables
# 		return grad_output.clone()
