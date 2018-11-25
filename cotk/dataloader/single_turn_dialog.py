from collections import Counter
from itertools import chain
import random

import numpy as np

from .dataloader import Dataloader

class SingleTurnDialog(Dataloader):
	def __init__(self):
		super().__init__()

		self.ext_vocab = ["<pad>", "<unk>", "<go>", "<eos>"]
		self.pad_id = 0
		self.unk_id = 1
		self.go_id = 2
		self.eos_id = 3
		self.key_name = ["train", "dev", "test"]

		self.data = {}
		self.index = {}
		self.batch_id = {}
		self.batch_size = {}

		self.word2id = {}
		self.vocab_list = []

		for key in self.key_name:
			self.batch_id[key] = 0
			self.batch_size[key] = None

	@property
	def vocab_size(self):
		return len(self.vocab_list)

	def restart(self, key, batch_size=None, shuffle=True):
		if not shuffle:
			if batch_size is None and self.batch_size[key] is None:
				raise ValueError("You need batch_size to intialize.")
			if self.batch_size[key] is not None and \
					batch_size is not None and batch_size != self.batch_size[key]:
				raise ValueError("If you want to change batch_size, you must shuffle.")
		if shuffle and batch_size is None:
			raise ValueError("You need batch_size to shuffle.")
		if shuffle:
			random.shuffle(self.index[key])

		self.batch_id[key] = 0
		if batch_size is not None:
			self.batch_size[key] = batch_size
		print("%s set restart, %d batches and %d left" % (key, \
				len(self.index[key]) // self.batch_size[key], \
				len(self.index[key]) % self.batch_size[key]))

	def get_batch(self, key, batch_id):
		st, ed = batch_id * self.batch_size[key], (batch_id + 1) * self.batch_size[key]
		index = self.index[key][st:ed]
		if ed > len(self.index[key]):
			return None
		res = {}
		res["post_length"] = np.array(list(map(lambda i: len(self.data[key]['post'][i]), index)))
		res["resp_length"] = np.array(list(map(lambda i: len(self.data[key]['resp'][i]), index)))
		res["post"] = np.zeros((self.batch_size[key], np.max(res["post_length"])), dtype=int)
		res["resp"] = np.zeros((self.batch_size[key], np.max(res["resp_length"])), dtype=int)
		for i, j in enumerate(index):
			post = self.data[key]['post'][j]
			resp = self.data[key]['resp'][j]
			res["post"][i, :len(post)] = post
			res["resp"][i, :len(resp)] = resp
		return res

	def get_next_batch(self, key):
		res = self.get_batch(key, self.batch_id[key])
		self.batch_id[key] += 1
		return res

	def sen_to_index(self, sen):
		return list(map(lambda word: self.word2id[word], sen))

	def index_to_sen(self, index, trim=True):
		if trim:
			try:
				index = index[:index.index(self.pad_id)]
			except ValueError:
				pass
		return list(map(lambda word: self.vocab_list[word], index))


class OpenSubtitles(SingleTurnDialog):
	def __init__(self, file_path, min_vocab_times=10, max_sen_length=50):
		super(OpenSubtitles, self).__init__()

		origin_data = {}
		for key in self.key_name:
			f = open("%s/opensub_pair_%s.post" % (file_path, key))
			g = open("%s/opensub_pair_%s.response" % (file_path, key))
			origin_data[key] = {}
			origin_data[key]['post'] = list(map(lambda line: line.split(), f.readlines()))
			origin_data[key]['resp'] = list(map(lambda line: line.split(), g.readlines()))

		vocab = list(chain(*(origin_data['train']['post'] + origin_data['train']['resp'])))
		left_vocab = list(filter(lambda x: x[1] >= min_vocab_times, Counter(vocab).most_common()))
		self.vocab_list = self.ext_vocab + list(map(lambda x: x[0], left_vocab))
		self.word2id = {w: i for i, w in enumerate(self.vocab_list)}
		print("vocab list length = %d" % len(self.vocab_list))

		line2id = lambda line: ([self.go_id] + \
					list(map(lambda word: self.word2id[word] if word in self.word2id else self.unk_id, line)) + \
					[self.eos_id])[:max_sen_length]

		for key in self.key_name:
			self.data[key] = {}

			self.data[key]['post'] = list(map(line2id, origin_data[key]['post']))
			self.data[key]['resp'] = list(map(line2id, origin_data[key]['resp']))
			vocab = list(chain(*(origin_data[key]['post'] + origin_data[key]['resp'])))
			vocab_num = len(vocab)
			OOV_num = len(list(filter(lambda word: word not in self.word2id, vocab)))
			length = list(map(len, origin_data[key]['post'] + origin_data[key]['resp']))
			cut_num = np.sum(np.maximum(np.array(length) - max_sen_length + 1, 0))
			print("%s set. OOV rate: %f, max length before cut: %d, cut word rate: %f" % \
					(key, OOV_num / vocab_num, max(length), cut_num / vocab_num))

		for key in self.key_name:
			self.index[key] = list(range(len(self.data[key]['post'])))
