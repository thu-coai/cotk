# coding:utf-8
import logging
import time
import os

import torch
from torch import nn, optim

from utils import Storage, cuda, BaseModel, SummaryHelper, getMean
from network import Network

class Seq2seq(BaseModel):
	def __init__(self, param):
		args = param.args
		net = Network(param)
		self.optimizer = optim.Adam(net.get_parameters_by_name(), lr=args.lr)
		optimizerList = {"optimizer": self.optimizer}

		super().__init__(param, net, optimizerList)

		self.create_summary()

	def create_summary(self):
		self.summaryHelper = SummaryHelper("tensorboard/%s_%s" % \
				(self.param.args.logname, time.strftime("%H%M%S", time.localtime())))

		self.trainSummary = self.summaryHelper.addGroup(\
			scalar=["loss", "word_loss", "perplexity"],\
			prefix="gen")

		scalarlist = ["word_loss", "perplexity"]
		tensorlist = []
		textlist = []
		emblist = []
		for i in self.args.show_sample:
			textlist.append("show_str%d" % i)
		self.devSummary = self.summaryHelper.addGroup(\
			scalar=scalarlist,\
			tensor=tensorlist,\
			text=textlist,\
			embedding=emblist,\
			prefix="dev")
		self.testSummary = self.summaryHelper.addGroup(\
			scalar=scalarlist,\
			tensor=tensorlist,\
			text=textlist,\
			embedding=emblist,\
			prefix="test")

	def _preprocess_batch(self, data):
		incoming = Storage()
		incoming.data = data = Storage(data)
		data.batch_size = data.post.shape[0]
		data.post = cuda(torch.LongTensor(data.post.transpose(1, 0))) # length * batch_size
		data.resp = cuda(torch.LongTensor(data.resp.transpose(1, 0))) # length * batch_size
		return incoming

	def get_next_batch(self, dm, key, restart=True):
		data = dm.get_next_batch(key)
		if data is None:
			if restart:
				dm.restart(key)
				return self.get_next_batch(dm, key, False)
			else:
				return None
		return self._preprocess_batch(data)

	def get_select_batch(self, dm, key, i):
		data = dm.get_batch(key, i)
		if data is None:
			return None
		return self._preprocess_batch(data)

	def train(self, batch_num):
		args = self.param.args
		dm = self.param.volatile.dm
		datakey = 'train'

		for _ in range(batch_num):
			self.now_batch += 1
			incoming = self.get_next_batch(dm, datakey)
			incoming.args = Storage()

			self.optimizer.zero_grad()
			self.net.forward(incoming)

			loss = incoming.result.loss
			self.trainSummary(self.now_batch, self.convertGPUtoCPU(incoming.result))
			logging.info("batch %d : gen loss=%f", self.now_batch, loss.detach().cpu().numpy())

			loss.backward()
			nn.utils.clip_grad_norm_(self.net.parameters(), args.grad_clip)
			self.optimizer.step()

	def evaluate(self, key):
		args = self.param.args
		dm = self.param.volatile.dm

		dm.restart(key, args.batch_size, shuffle=False)

		result_arr = []
		while True:
			incoming = self.get_next_batch(dm, key, restart=False)
			if incoming is None:
				break
			incoming.args = Storage()

			with torch.no_grad():
				self.net.forward(incoming)
			result_arr.append(incoming.result)

		detail_arr = Storage()
		for i in args.show_sample:
			incoming = self.get_select_batch(dm, key, i)
			incoming.args = Storage()
			with torch.no_grad():
				self.net.detail_forward(incoming)
			detail_arr["show_str%d" % i] = incoming.result.show_str

		detail_arr.update({key:getMean(result_arr, key) for key, _ in result_arr[0].items()})
		return detail_arr

	def train_process(self):
		args = self.param.args
		dm = self.param.volatile.dm

		while self.now_epoch < args.epochs:
			self.now_epoch += 1
			self.updateOtherWeights()

			dm.restart('train', args.batch_size)
			self.net.train()
			self.train(args.batch_per_epoch)

			self.net.eval()
			devloss_detail = self.evaluate("dev")
			self.devSummary(self.now_batch, devloss_detail)
			logging.info("epoch %d, evaluate dev", self.now_epoch)

			testloss_detail = self.evaluate("test")
			self.testSummary(self.now_batch, testloss_detail)
			logging.info("epoch %d, evaluate test", self.now_epoch)

			self.save_checkpoint()

	def test(self, key):
		args = self.param.args
		dm = self.param.volatile.dm
		
		dm.restart(key, 1, shuffle=False)
		result_arr = []
		while True:
			incoming = self.get_next_batch(dm, key, restart=False)
			if incoming is None:
				break
			incoming.args = Storage()
			with torch.no_grad():
				self.net.forward(incoming)
			result_arr.append(incoming.result)

		dm.restart(key, 1, shuffle=False)
		detail_arr = []
		while True:
			incoming = self.get_next_batch(dm, key, restart=False)
			if incoming is None:
				break
			incoming.args = Storage()
			with torch.no_grad():
				self.net.detail_forward(incoming)
			detail_arr.append(incoming.result)
		show_metric = ["perplexity"]

		logging.info("%s Test Result:", key)
		for s in show_metric:
			logging.info("\t%s:\t%f", s, getMean(result_arr, s))

		if not os.path.exists(args.outdir):
			os.makedirs(args.outdir)
		filename = args.outdir + "/%s.txt" % key
		with open(filename, 'w') as f:
			for i in detail_arr:
				f.write(i.show_str)
			f.flush()
		logging.info("result output to %s.", filename)

	def test_process(self):
		logging.info("Test Start.")
		self.net.eval()
		self.test("dev")
		self.test("test")
		logging.info("Test Finish.")
