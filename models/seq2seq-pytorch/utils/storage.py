# coding:utf-8

import logging

class Storage(dict):
	def __init__(self, *args, **kw):
		dict.__init__(self, *args, **kw)

	def __getattr__(self, key):
		if key in self:
			return self[key]
		else:
			return getattr(super(Storage, self), key)

	def __setattr__(self, key, value):
		self[key] = value

	def __delattr__(self, key):
		del self[key]

	def compare(self, newobj):
		for i, j in newobj.items():
			if i not in self:
				logging.info("%s new: %s", i, j)
			elif isinstance(self[i], Storage):
				self[i].compare(j)
			else:
				if self[i] != j:
					logging.info("%s old: %s; new: %s", i, self[i], j)

	def join(self, commonobj):
		for i, j in commonobj.items():
			if i not in self:
				self[i] = j
			elif isinstance(self[i], Storage):
				self[i].join(j)
			else:
				if self[i] != j:
					logging.info("args: %s common: %s; new: %s", i, j, self[i])

	def check(self, checkobj):
		for i, j in checkobj.items():
			if i not in self:
				logging.info("args not find: %s", i)
				exit()
			elif isinstance(self[i], Storage):
				self[i].check(j)
