import torch
import numpy as np

from .cuda_helper import cuda, Tensor

def getMMD(x, y, same=False, theta=1):
	# num * hidden
	dis = torch.norm(x.unsqueeze(1) - y.unsqueeze(0), p=2, dim=2) / theta
	allnum = x.shape[0] * y.shape[0]
	if same:
		dis = dis + cuda(torch.eye(x.shape[0])) * 1e9
		allnum -= x.shape[0]
	return (-dis**2 / 2).exp().sum() / allnum

def gaussMMD(x, mean=0, std=1, sample=None, theta=1):
	if sample is None:
		sample = x.shape[0]
	samples = Tensor(np.random.randn(sample, x.shape[1]) * std + mean)
	return getMMD(x, x, True, theta) + getMMD(samples, samples, True, theta) - 2 * getMMD(x, samples, theta=theta)
