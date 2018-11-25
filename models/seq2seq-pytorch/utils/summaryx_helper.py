# coding:utf-8

import os
import os.path

import tensorboardX as tx
import torch

class SummaryHelper:
	def __init__(self, filename):
		directory = os.path.dirname(filename)
		if not os.path.exists(directory):
			os.makedirs(directory)
		self.writer = tx.SummaryWriter(filename)

	def addGroup(self, *, scalar=[], tensor=[], image=[], text=[], embedding=[], prefix=""):

		def write(i, data):
			for name in scalar:
				if name in data:
					self.writer.add_scalar("%s/%s" % (prefix, name), data[name], i)
			for name in tensor:
				if name in data:
					self.writer.add_histogram("%s/%s" % (prefix, name), data[name], i)
			for name in image:
				if name in data:
					self.writer.add_image("%s/%s" % (prefix, name), data[name], i)
			for name in text:
				if name in data:
					self.writer.add_text("%s/%s" % (prefix, name), data[name], i)
			for name in embedding:
				if name in data:
					emb = data[name]
					metadata = []
					valuedata = []
					for tag, value in emb.items():
						metadata += [tag] * value.shape[0]
						valuedata.append(value)
					self.writer.add_embedding(torch.cat(valuedata, dim=0), \
						metadata=metadata, global_step=i, tag="%s/%s" % (prefix, name))

		return write
