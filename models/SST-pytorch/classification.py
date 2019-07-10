# coding:utf-8
import logging
import time
import os

import torch
from torch import nn, optim
import numpy as np
import tqdm

from utils import Storage, cuda, BaseModel, SummaryHelper, get_mean, storage_to_list, \
	CheckpointManager, LongTensor, get_accuracy
from network import Network

class Classification(BaseModel):
	def __init__(self, param):
		args = param.args
		net = Network(param)
		self.optimizer = optim.Adam(net.get_parameters_by_name(), lr=args.lr)
		optimizerList = {"optimizer": self.optimizer}
		checkpoint_manager = CheckpointManager(args.name, args.model_dir, \
						args.checkpoint_steps, args.checkpoint_max_to_keep, "min")
		super().__init__(param, net, optimizerList, checkpoint_manager)

		self.create_summary()

	def create_summary(self):
		args = self.param.args
		self.summaryHelper = SummaryHelper("%s/%s_%s" % \
				(args.log_dir, args.name, time.strftime("%H%M%S", time.localtime())), \
				args)
		self.trainSummary = self.summaryHelper.addGroup(\
			scalar=["loss", "accuracy_on_batch"],\
			prefix="train")

		scalarlist = ["loss", "accuracy"]
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
		data.batch_size = data.sent.shape[0]
		data.sent = cuda(torch.LongTensor(data.sent.transpose(1, 0))) # length * batch_size
		data.label = cuda(torch.LongTensor(data.label))
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

	def get_batches(self, dm, key):
		batches = list(dm.get_batches(key, batch_size=self.args.batch_size, shuffle=False))
		return len(batches), (self._preprocess_batch(data) for data in batches)

	def get_select_batch(self, dm, key, i):
		data = dm.get_batch(key, i)
		if data is None:
			return None
		return self._preprocess_batch(data)

	def train(self, batch_num):
		args = self.param.args
		dm = self.param.volatile.dm
		datakey = 'train'

		for i in range(batch_num):
			self.now_batch += 1
			incoming = self.get_next_batch(dm, datakey)
			incoming.args = Storage()

			if (i+1) % args.batch_num_per_gradient == 0:
				self.zero_grad()
			self.net.forward(incoming)

			loss = incoming.result.loss
			accuracy = np.mean((incoming.result.label == incoming.result.prediction).float().detach().cpu().numpy())
			detail_arr = storage_to_list(incoming.result)
			detail_arr.update({'accuracy_on_batch': accuracy})
			self.trainSummary(self.now_batch, detail_arr)
			logging.info("batch %d : classification loss=%f, batch accuracy=%f", \
				self.now_batch, loss.detach().cpu().numpy(), accuracy)

			loss.backward()

			if (i+1) % args.batch_num_per_gradient == 0:
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
			index = [i * args.batch_size + j for j in range(args.batch_size)]
			incoming = self.get_select_batch(dm, key, index)
			incoming.args = Storage()
			with torch.no_grad():
				self.net.forward(incoming)
			detail_arr["show_str%d" % i] = incoming.result.show_str

		detail_arr.update({'loss':get_mean(result_arr, 'loss'), \
			'accuracy':get_accuracy(result_arr, label_key='label', prediction_key='prediction')})
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

			self.save_checkpoint(value=devloss_detail.loss.tolist())

	def test(self, key):
		args = self.param.args
		dm = self.param.volatile.dm

		metric = dm.get_accuracy_metric()
		batch_num, batches = self.get_batches(dm, key)
		logging.info("eval accuracy")
		for incoming in tqdm.tqdm(batches, total=batch_num):
			incoming.args = Storage()
			with torch.no_grad():
				self.net.forward(incoming)
			data = incoming.data
			data.prediction = imcoming.result.prediction
			data.label = imcoming.data.label
			metric.forward(data)
		res = metric.close()

		if not os.path.exists(args.out_dir):
			os.makedirs(args.out_dir)
		filename = args.out_dir + "/%s_%s.txt" % (args.name, key)

		with open(filename, 'w') as f:
			logging.info("%s Test Result:", key)
			for key, value in res.items():
				if isinstance(value, float) or isinstance(value, bytes):
					logging.info("\t{}:\t{}".format(key, value))
					f.write("{}:\t{}\n".format(key, value))
			f.flush()
		logging.info("result output to %s.", filename)

	def test_process(self):
		logging.info("Test Start.")
		self.net.eval()
		self.test("dev")
		self.test("test")
		logging.info("Test Finish.")
