# coding:utf-8
import logging

import torch
from torch import nn

from utils import zeros, LongTensor,\
			BaseNetwork, MyGRU, Storage, gumbel_max, flattenSequence, SingleAttnGRU, SequenceBatchNorm

# pylint: disable=W0221
class Network(BaseNetwork):
	def __init__(self, param):
		super().__init__(param)

		self.embLayer = EmbeddingLayer(param)
		self.encoder = Encoder(param)
		self.predictionNetwork = PredictionNetwork(param)

	def forward(self, incoming):
		incoming.result = Storage()

		self.embLayer.forward(incoming)
		self.encoder.forward(incoming)
		self.predictionNetwork.forward(incoming)

		incoming.result.loss = incoming.result.classification_loss

		if torch.isnan(incoming.result.loss).detach().cpu().numpy() > 0:
			logging.info("Nan detected")
			logging.info(incoming.result)
			raise FloatingPointError("Nan detected")

class EmbeddingLayer(nn.Module):
	def __init__(self, param):
		super().__init__()
		self.args = args = param.args
		self.param = param
		volatile = param.volatile

		self.embLayer = nn.Embedding(volatile.dm.vocab_size, args.embedding_size)
		self.embLayer.weight = nn.Parameter(torch.Tensor(volatile.wordvec))

	def forward(self, incoming):
		'''
		inp: data
		output: post
		'''
		incoming.sent = Storage()
		incoming.sent.embedding = self.embLayer(incoming.data.sent)
		incoming.sent.embLayer = self.embLayer

class Encoder(nn.Module):
	def __init__(self, param):
		super().__init__()
		self.args = args = param.args
		self.param = param

		self.sentGRU = MyGRU(args.embedding_size, args.eh_size, bidirectional=True)
		self.drop = nn.Dropout(args.droprate)
		if self.args.batchnorm:
			self.seqnorm = SequenceBatchNorm(args.eh_size * 2)
			self.batchnorm = nn.BatchNorm1d(args.eh_size * 2)

	def forward(self, incoming):
		incoming.hidden = hidden = Storage()
		hidden.h_n, hidden.h = self.sentGRU.forward(incoming.sent.embedding, incoming.data.sent_length, need_h=True)
		if self.args.batchnorm:
			hidden.h = self.seqnorm(hidden.h, incoming.data.sentGRU_length)
			hidden.h_n = self.batchnorm(hidden.h_n)
		hidden.h = self.drop(hidden.h)
		hidden.h_n = self.drop(hidden.h_n)

class PredictionNetwork(nn.Module):
	def __init__(self, param):
		super().__init__()
		self.args = args = param.args
		self.param = param
		self.predictionLayer = nn.Sequential(nn.Linear(args.eh_size * 2, args.eh_size),
											nn.ReLU(),
											nn.Linear(args.eh_size, args.class_num))

		self.lossCE = nn.CrossEntropyLoss()

	def forward(self, incoming):

		dm = self.param.volatile.dm
		logit = self.predictionLayer(incoming.hidden.h_n)
		incoming.result.classification_loss = self.lossCE(logit, incoming.data.label)
		incoming.result.prediction = prediction = torch.max(logit, dim=1)[1]
		incoming.result.label = incoming.data.label
		incoming.result.sent_str = sent_str = \
				[" ".join(dm.convert_ids_to_tokens(incoming.data.sent[:, i].tolist())) \
						for i in range(incoming.data.batch_size)]
		incoming.result.show_str = "\n".join(["sentence: " + a + "\n" + "prediction: " + str(b) + "\n" + \
				"golden: " + str(c) + "\n" \
				for a, b, c in zip(sent_str, prediction, incoming.data.label)])
