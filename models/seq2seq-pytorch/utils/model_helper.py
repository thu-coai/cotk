import logging
import time
import os
import shutil
import json

import torch
import numpy as np

from .cuda_helper import cuda, Tensor
from .anneal_helper import AnnealHelper, AnnealParameter
from .storage import Storage

class BaseModel():
	def __init__(self, param, net, optimizerList, checkpoint_manager, helperList=None):
		self.param = param
		self.args = args = param.args
		self.param.other_weights = Storage()

		self.net = net

		_ = list(self.net.get_parameters_by_name())
		self.optimizerList = optimizerList

		self.now_batch = 0
		self.now_epoch = 0

		self.checkpoint_manager = checkpoint_manager
		self.helperList = helperList or {}
		self.helperList["checkpoint_manager"] = self.checkpoint_manager

		self.anneal_list = []
		for key, v in args.items():
			if isinstance(v, AnnealParameter):
				if v[0] == "hold":
					self.param.other_weights[key] = v[1]["value"]
				elif v[0] == "anneal":
					self.anneal_list.append(AnnealHelper(self, key, **v[1]))
					self.param.other_weights[key] = v[1]["beginValue"]

		if args.cuda:
			logging.info("initializing cuda")
			Tensor(1)
			logging.info("cuda initialized")

		if args.restore is not None:
			checkpoint = self.checkpoint_manager.restore(args.restore)
			diff = args - checkpoint["args"]
			if diff:
				logging.info("Args differences\n%s", json.dumps(diff, indent=2))
			self.now_batch = checkpoint['now_batch']
			self.now_epoch = checkpoint['now_epoch']
			self.net.load_state_dict(checkpoint['weights'], param.volatile.load_exclude_set)
			self.param.other_weights = checkpoint['other_weights']
			for name, optimizer in self.optimizerList.items():
				if checkpoint[name]['state'] and self.param.args.restore_optimizer:
					optimizer.load_state_dict(checkpoint[name])
					self.optimizerCuda(optimizer)
			for name, helper in self.helperList.items():
				helper.load_state_dict(checkpoint[name])
			logging.info("loaded checkpoint at %d epochs, %d batches", self.now_epoch, self.now_batch)

		for key, v in args.items():
			if isinstance(v, AnnealParameter):
				if v[0] == "set":
					self.param.other_weights[key] = v[1]["value"]
				elif v[0] == "set&anneal":
					self.anneal_list.append(AnnealHelper(self, key, 0, 0, **v[1]))
					self.param.other_weights[key] = v[1]["startValue"]

		if args.restore is not None and param.volatile.restoreCallback:
			param.volatile.restoreCallback(self)

		cuda(self.net)

	def optimizerCuda(self, optimizer):
		for state in optimizer.state.values():
			for k, v in state.items():
				if torch.is_tensor(v):
					state[k] = cuda(v)

	def updateOtherWeights(self):
		for a in self.anneal_list:
			a.step()

	def updateOver(self):
		for a in self.anneal_list:
			if not a.over():
				return False
		return True

	def zero_grad(self):
		for p in self.net.parameters():
			if p.grad is not None:
				p.grad.detach_()
				p.grad.zero_()

	def save_checkpoint(self, value=None, filename=None):
		args = self.args
		if filename is None:
			filename = "%s_%s" % (self.param.args.name, \
					time.strftime("%Y%m%d_%H%M%S", time.localtime()))
		state = {\
			'now_epoch': self.now_epoch,\
			'now_batch': self.now_batch,\
			'args': self.param.args,\
			'weights': self.net.state_dict(),\
			'other_weights': self.param.other_weights,\
		}
		for name, optimizer in self.optimizerList.items():
			state[name] = optimizer.state_dict()
		for name, helper in self.helperList.items():
			state[name] = helper.state_dict()
		self.checkpoint_manager.save(state, filename, value)

	def checkgrad(self):
		logging.info("checkgrad:")
		for name, p in self.net.named_parameters():
			if p.grad is not None and p.abs().sum().tolist() > 0:
				logging.info("\t%s", name)

def get_mean(loss_arr, key):
	if key in loss_arr[0]:
		return np.mean(list(map(lambda x: x[key].detach().cpu().numpy(), loss_arr)))
	else:
		return 0

def storage_to_list(incoming):
	for i, j in incoming.items():
		if "tolist" in dir(j):
			incoming[i] = j.tolist()
	return incoming


