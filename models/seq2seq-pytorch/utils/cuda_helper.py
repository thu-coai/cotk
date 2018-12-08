# -*- coding: utf-8 -*-

import torch

use_cuda = False
device_num = 0
_zeros = None
_ones = None
_tensor_type = None
_long_tensor_type = None
_cuda = None

def cpuzeros(*size):
	return torch.FloatTensor(*size).fill_(0)
def cpuones(*size):
	return torch.FloatTensor(*size).fill_(1)

def gpuzeros(*size):
	with torch.cuda.device(device_num):
		return torch.cuda.FloatTensor(*size).fill_(0)
def gpuones(*size):
	with torch.cuda.device(device_num):
		return torch.cuda.FloatTensor(*size).fill_(1)

# pylint: disable=W0603
def init(device=None, __cuda=False):
	global _cuda, _zeros, _ones, use_cuda, device_num, _tensor_type, _long_tensor_type
	if __cuda:
		use_cuda = True
		device_num = device
		torch.cuda.set_device(device_num)
		_cuda = lambda x: x.cuda(device_num)
		_zeros = gpuzeros
		_ones = gpuones
		_tensor_type = torch.cuda.FloatTensor
		_long_tensor_type = torch.cuda.LongTensor
	else:
		_cuda = lambda x: x
		_zeros = cpuzeros
		_ones = cpuones
		_tensor_type = torch.FloatTensor
		_long_tensor_type = torch.LongTensor

def Tensor(*x):
	return _tensor_type(*x)

def LongTensor(*x):
	return _long_tensor_type(*x)

def cuda(*x):
	return _cuda(*x)

def zeros(*size):
	return _zeros(*size)

def ones(*size):
	return _ones(*size)
