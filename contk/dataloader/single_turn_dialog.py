from collections import Counter
from itertools import chain
import random

import numpy as np

from .dataloader import Dataloader

class SingleTurnDialog(Dataloader):
	r"""Base class for single-turn dialog datasets.

	Attributes:
		ext_vocab (list): special tokens, must be placed at beginning of `vocab_list`.
			default: ``["<pad>", "<unk>", "<go>", "<eos>"]``
		pad_id (int): token for padding. default: 0
		unk_id (int): token for unkown words. default: 1
		go_id (int): token at the beginning of sentences. default: 2
		eos_id (int): token at the end of sentences. default: 3
		key_name (list): name of subsets of the data. default: ``["train", "dev", "test"]``
		vocab_list (list): all tokens of the datasets.
		word2id (dict):  a dict mapping tokens to index.
			Maybe you want to use :meth:`.sen_to_index` instead.

	Note:
		(For developer) You must initialize following attributes in the subclasses.

		* vocab (list): don't forget `ext_vocab`.
		* word2id (dict): should be initialized by ``{w: i for i, w in enumerate(self.vocab_list)}``
		* data (dict): a dict mapping `key_name` to lists, which contains sentences in index form.
		* index (dict): a dict mapping `key_name` to lists, should be initialized by ``list(range(len(data[key])))``
	"""
	def __init__(self, _key_name=None):
		super().__init__()

		# initialize by default value. (can be overwritten by subclass)
		self.ext_vocab = ["<pad>", "<unk>", "<go>", "<eos>"]
		self.pad_id = 0
		self.unk_id = 1
		self.go_id = 2
		self.eos_id = 3
		if not _key_name:
			_key_name = ["train", "dev", "test"]
		self.key_name = _key_name

		# initialize by subclass
		self.vocab_list = []
		self.word2id = {}
		self.data = {}
		self.index = {}

		# don't need initialize
		self.batch_id = {}
		self.batch_size = {}
		for key in self.key_name:
			self.batch_id[key] = 0
			self.batch_size[key] = None

	@property
	def vocab_size(self):
		'''Equals to len(self.vocab_list). Read only.
		'''
		return len(self.vocab_list)

	def restart(self, key, batch_size=None, shuffle=True):
		'''Initialize mini-batches. Must call this function before :func:`get_next_batch`
		or an epoch is end.

		Arguments:
			key (str): must be contained in `key_name`
			batch_size (None or int): default (None): use last batch_size.
			shuffle (bool): whether to shuffle the data. default: ``True``
		'''
		if key not in self.key_name:
			raise ValueError("No set named %s." % key)
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

	def get_batch(self, key, index):
		'''Get a batch of specified `index`.```

		Arguments:
			key (str): must be contained in `key_name`
			index (list): a list of specified index

		Returns:
			A dict at least contains ``post``, ``post_length``, ``resp``,
			``resp_length``. See the example belows.

		Examples:
			>>> dataloader.get_batch('train', 1)
			>>> 

		Todo:
			* fix the missing example
		'''
		if key not in self.key_name:
			raise ValueError("No set named %s." % key)
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
		'''Get next batch.

		Arguments:
			key (str): must be contained in `key_name`

		Returns:
			A dict like :func:.get_batch, or None if the epoch is end.
		'''
		if key not in self.key_name:
			raise ValueError("No set named %s." % key)
		batch_id = self.batch_id[key]
		start, end = batch_id * self.batch_size[key], (batch_id + 1) * self.batch_size[key]
		if end > len(self.index[key]):
			return None
		index = self.index[key][start:end]
		res = self.get_batch(key, index)
		self.batch_id[key] += 1
		return res

	def sen_to_index(self, sen):
		'''Convert a sentences from string to index representation.

		Arguments:
			sen (list): a list of str, representing each token of the sentences.

		Examples:
			>>> dataloader.sen_to_index(
			...		["<go>", "I", "have", "been", "to", "Sichuan", "province", "eos"])
			>>> 

		Todo:
			* fix the missing example
		'''
		return list(map(lambda word: self.word2id.get(word, self.unk_id, sen)))

	def index_to_sen(self, index, trim=True):
		'''Convert a sentences from index to string representation

		Arguments:
			index (list): a list of int
			trim (bool): if True, ignore `pad_id` at the end of `index`.

		Examples:
			>>> dataloader.index_to_sen(
			...		[])
			>>> 

		Todo:
			* fix the missing example
		'''
		if trim:
			try:
				index = index[:index.index(self.pad_id)]
			except ValueError:
				pass
		return list(map(lambda word: self.vocab_list[word], index))


class OpenSubtitles(SingleTurnDialog):
	'''A dataloder for OpenSubtitles dataset. 

	Arguments:
		file_path (str): a str indicates the dir of OpenSubtitles dataset.
		min_vocab_times (int): A cut-off threshold of `UNK` tokens. All tokens appear
			less than `min_vocab_times`	will be replaced by ``<unk>``. Default: 10.
		max_sen_length (int): All sentences longer than `max_sen_length` will be shortened
			to first `max_sen_length` tokens. Default: 50.

	Arguments:
		Inherited from :class:.SingleTurnDialog

	Todo:
		* add references
	'''
	def __init__(self, file_path, min_vocab_times=10, max_sen_length=50):
		super(OpenSubtitles, self).__init__()

		origin_data = {}
		for key in self.key_name:
			f_file = open("%s/opensub_pair_%s.post" % (file_path, key))
			g_file = open("%s/opensub_pair_%s.response" % (file_path, key))
			origin_data[key] = {}
			origin_data[key]['post'] = list(map(lambda line: line.split(), f_file.readlines()))
			origin_data[key]['resp'] = list(map(lambda line: line.split(), g_file.readlines()))

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
			oov_num = len(list(filter(lambda word: word not in self.word2id, vocab)))
			length = list(map(len, origin_data[key]['post'] + origin_data[key]['resp']))
			cut_num = np.sum(np.maximum(np.array(length) - max_sen_length + 1, 0))
			print("%s set. OOV rate: %f, max length before cut: %d, cut word rate: %f" % \
					(key, oov_num / vocab_num, max(length), cut_num / vocab_num))

		for key in self.key_name:
			self.index[key] = list(range(len(self.data[key]['post'])))
