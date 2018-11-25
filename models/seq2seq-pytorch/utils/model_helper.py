import logging
import time
import os
import shutil

import torch
import numpy as np

from utils import cuda, TensorType, AnnealHelper

class BaseModel():
	def __init__(self, param, net, optimizerList):
		self.param = param
		self.args = args = param.args
		self.param.other_weights = {}

		self.net = net

		_ = list(self.net.get_parameters_by_name())
		self.optimizerList = optimizerList

		self.now_batch = 0
		self.now_epoch = 0
		self.best_loss = 1e100
		self.checkpoint_list = []

		self.anneal_list = []
		for key, v in args.items():
			if isinstance(v, tuple):
				if v[0] == "hold":
					self.param.other_weights[key] = v[1]["value"]
				elif v[0] == "anneal":
					self.anneal_list.append(AnnealHelper(self, key, **v[1]))
					self.param.other_weights[key] = v[1]["beginValue"]

		if args.cuda:
			logging.info("initializing cuda")
			TensorType()(1)
			logging.info("cuda initialized")

		if args.restore is not None:
			if args.restore == "last":
				if not os.path.isfile("model/checkpoint_list"):
					raise ValueError("No Last checkpoint found")
				args.restore = open("model/checkpoint_list").readline()

			if os.path.isfile("model/%s.model" % args.restore):
				logging.info("loading checkpoint %s", args.restore)
				checkpoint = torch.load("model/%s.model" % args.restore, \
						map_location=lambda storage, loc: storage)
				checkpoint["args"].compare(args)
				self.now_batch = checkpoint['now_batch']
				self.now_epoch = checkpoint['now_epoch']
				self.best_loss = checkpoint['best_loss']
				self.net.load_state_dict(checkpoint['weights'], args.load_exclude_set)
				self.param.other_weights = checkpoint['other_weights']
				for name, optimizer in self.optimizerList.items():
					if checkpoint[name]['state'] and self.param.args.restore_optimizer:
						optimizer.load_state_dict(checkpoint[name])
						self.optimizerCuda(optimizer)
				logging.info("loaded checkpoint at %d epochs, %d batchs", self.now_epoch, self.now_batch)
			else:
				logging.info("no checkpoint found at %s", args.restore)
				raise AssertionError("no checkpoint found")

		for key, v in args.items():
			if isinstance(v, tuple):
				if v[0] == "set":
					self.param.other_weights[key] = v[1]["value"]
				elif v[0] == "set&anneal":
					self.anneal_list.append(AnnealHelper(self, key, 0, 0, **v[1]))
					self.param.other_weights[key] = v[1]["startValue"]

		if args.restore is not None and args.restoreCallback:
			args.restoreCallback(self)
			del args['restoreCallback']

		cuda(self.net)

	def optimizerCuda(self, optimizer):
		for state in optimizer.state.values():
			for k, v in state.items():
				if torch.is_tensor(v):
					state[k] = cuda(v)

	def updateOtherWeights(self):
		for a in self.anneal_list:
			a.step()

	def save_checkpoint(self, is_best=False, filename=None):
		if filename is None:
			filename = "%s_%s" % (self.param.args.logname, \
					time.strftime("%Y%m%d_%H%M%S", time.localtime()))
		state = {\
			'now_epoch': self.now_epoch,\
			'now_batch': self.now_batch,\
			'best_loss': self.best_loss,\
			'args': self.param.args,\
			'weights': self.net.state_dict(),\
			'other_weights': self.param.other_weights,\
		}
		for name, optimizer in self.optimizerList.items():
			state[name] = optimizer.state_dict()
		if not os.path.exists("model"):
			os.makedirs("model")
		torch.save(state, "model/%s.model" % filename)

		open("model/checkpoint_list", "w").write(filename)

		if self.now_epoch % self.param.args.checkpoint_steps == 0:
			self.checkpoint_list.append(filename)
			if len(self.checkpoint_list) > self.param.args.checkpoint_max_to_keep:
				try:
					os.remove("model/%s.model" % self.checkpoint_list[0])
				except OSError:
					pass
				self.checkpoint_list.pop(0)
		else:
			if len(self.checkpoint_list) > 1:
				try:
					os.remove("model/%s.model" % self.checkpoint_list[-1])
				except OSError:
					pass
				self.checkpoint_list.pop()
			self.checkpoint_list.append(filename)

		if is_best:
			shutil.copyfile("model/%s.model" % filename, 'model/%s_best.model' % self.param.args.logname)

	def convertGPUtoCPU(self, incoming):
		for i, j in incoming.items():
			incoming[i] = j.detach().cpu().numpy()
		return incoming

	def checkgrad(self):
		logging.info("checkgrad:")
		for name, p in self.net.named_parameters():
			if p.grad is not None:
				logging.info("\t%s", name)

def getMean(loss_arr, key):
	if key in loss_arr[0]:
		return np.mean(list(map(lambda x: x[key].detach().cpu().numpy(), loss_arr)))
	else:
		return 0
