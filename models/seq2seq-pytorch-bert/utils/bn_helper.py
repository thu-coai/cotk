#coding: utf-8
import math
import numpy as np
import torch
from torch import nn
from torch.nn import Parameter

from .cuda_helper import zeros, Tensor, LongTensor
from .gumbel import gumbel_max
from .storage import Storage


class SequenceBatchNorm(nn.Module):
	def __init__(self, num_features):
		# seqlen * batch * XXXXX * num_features
		super().__init__()

		self.num_features = num_features
		self.bn = nn.BatchNorm1d(num_features)

	def forward(self, incoming, length):
		incoming_shape = incoming.shape
		seqlen = incoming_shape[0]
		batch_num = incoming_shape[1]
		assert self.num_features == incoming_shape[-1]
		assert len(length) == incoming_shape[1]

		incoming = incoming.reshape(seqlen, batch_num, -1, self.num_features)

		arr = []
		for i, l in enumerate(length):
			arr.append(incoming[:l, i])
		alllen = np.sum(length)
		incoming = torch.cat(arr, dim=0)

		incoming = self.bn(incoming.view(-1, self.num_features)).view(alllen, -1, self.num_features)

		#arr = []
		now = 0
		other_dim = incoming.shape[-2]
		res = zeros(seqlen, batch_num, other_dim, self.num_features)
		for i, l in enumerate(length):
			#arr.append(torch.cat([incoming[now:now+l], zeros(seqlen-l, other_dim, self.num_features)], dim=0))
			res[:l, i] = incoming[now:now+l]
			now += l
		#incoming = torch.stack(arr, 1)
		return res.view(*incoming_shape)
