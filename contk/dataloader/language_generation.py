"""Dataloader for language generation"""
from collections import Counter
from itertools import chain
import random

import numpy as np

from .dataloader import Dataloader
from ..metric import MetricChain, PerlplexityMetric, BleuCorpusMetric, LanguageGenerationRecorder

from .._utils import trim_before_target


class LanguageGeneration(Dataloader):
	r"""Base class for language generation datasets. This is an abstract class.

	Arguments:
			ext_vocab (list): special tokens. default: `["<pad>", "<unk>", "<go>", "<eos>"]`
			key_name (list): name of subsets of the data. default: `["train", "dev", "test"]`

	Attributes:
			ext_vocab (list): special tokens, be placed at beginning of `vocab_list`.
					For example: `["<pad>", "<unk>", "<go>", "<eos>"]`
			pad_id (int): token for padding, always equal to `0`
			unk_id (int): token for unkown words, always equal to `1`
			go_id (int): token at the beginning of sentences, always equal to `2`
			eos_id (int): token at the end of sentences, always equal to `3`
			key_name (list): name of subsets of the data. For example: `["train", "dev", "test"]`
			vocab_list (list): vocabulary list of the datasets.
			word2id (dict): a dict mapping tokens to index.
					Maybe you want to use :meth:`sen_to_index` instead.
	"""

	def __init__(self,
				 ext_vocab=None,
				 key_name=None,
				 ):
		super().__init__()

		# initialize by default value. (can be overwritten by subclass)
		self.ext_vocab = ext_vocab or ["<pad>", "<unk>", "<go>", "<eos>"]
		self.pad_id = self.ext_vocab.index("<pad>")
		self.unk_id = self.ext_vocab.index("<unk>")
		self.go_id = self.ext_vocab.index("<go>")
		self.eos_id = self.ext_vocab.index("<eos>")
		self.key_name = key_name or ["train", "dev", "test"]

		# initialize by subclass
		self.vocab_list, self.data = self._load_data()
		self.word2id = {w: i for i, w in enumerate(self.vocab_list)}

		# postprocess initialization
		self.index = {}
		self.batch_id = {}
		self.batch_size = {}
		for key in self.key_name:
			self.batch_id[key] = 0
			self.batch_size[key] = None
			self.index[key] = list(range(len(self.data[key])))

	def _load_data(self):
		r'''This function is called during the initialization.

		Returns:
				(tuple): tuple containing (refer to the following example):

						vocab_list (list): vocabulary list of the datasets.
						data (dict): a dict contains data.

		Examples:
		.. highlight:: python
		.. code-block:: python
				vocab_list = ["<pad>", "<unk>", "<go>", "<eos>", "how", \
										  "are", "you", "hello", "i", "am", \
										  "fine"]
				data = {
						"train": [
								[2, 5, 6, 7, 3],  # first sentence: <go> how are you <eos>
								[2, 9, 10, 11, 3], # second response: <go> i am fine <eos>
						],
						"dev": [...],   # similar to train
						"test": [...],  # similar to train
				}

		Notes:
				You can use ``ext_vocab``, ``key_name``, ``pad_id``, ``unk_id``, ``go_id``,
				``eos_id``, but other attributes are not initialized.
		'''
		raise NotImplementedError(
			"This function should be implemented by subclasses.")

	@property
	def vocab_size(self):
		'''Equals to `len(self.vocab_list)`. Read only.
		'''
		return len(self.vocab_list)

	def restart(self, key, batch_size=None, shuffle=True):
		'''Initialize mini-batches. Must call this function before :func:`get_next_batch`
		or an epoch is end.

		Arguments:
				key (str): must be contained in `key_name`
				batch_size (None or int): default (None): use last batch_size.
				shuffle (bool): whether to shuffle the data. default: `True`
		'''
		if key not in self.key_name:
			raise ValueError("No set named %s." % key)
		if batch_size is None and self.batch_size[key] is None:
			raise ValueError("You need batch_size to intialize.")
		if shuffle:
			random.shuffle(self.index[key])

		self.batch_id[key] = 0
		if batch_size is not None:
			self.batch_size[key] = batch_size
		print("%s set restart, %d batches and %d left" %
			  (key, len(self.index[key]) //
			   self.batch_size[key], len(self.index[key]) %
			   self.batch_size[key]))

	def get_batch(self, key, index):
		'''Get a batch of specified `index`.

		Arguments:
				key (str): must be contained in `key_name`
				index (list): a list of specified index

		Returns:
				A dict at least contains ``sentence``, ``sentence_length``. See the example belows.

		Examples:
				>>> dataloader.get_batch('train', 1)
				>>>

		Todo:
				* fix the missing example
		'''
		if key not in self.key_name:
			raise ValueError("No set named %s." % key)
		res = {}
		batch_size = len(index)
		res["sentence_length"] = np.array(
			list(map(lambda i: len(self.data[key][i]), index)))
		res["sentence"] = np.zeros(
			(batch_size, np.max(res["sentence_length"])), dtype=int)
		for i, j in enumerate(index):
			sentence = self.data[key][j]
			res["sentence"][i, :len(sentence)] = sentence
		return res

	def get_next_batch(self, key, ignore_left_samples=False):
		'''Get next batch.

		Arguments:
				key (str): must be contained in `key_name`
				ignore_left_samples (bool): Ignore the last batch, whose sample num
						is not equal to `batch_size`. Default: `False`

		Returns:
				A dict like :func:`get_batch`, or None if the epoch is end.
		'''
		if key not in self.key_name:
			raise ValueError("No set named %s." % key)
		if self.batch_size[key] is None:
			raise RuntimeError(
				"Please run restart before calling this function.")
		batch_id = self.batch_id[key]
		start, end = batch_id * \
			self.batch_size[key], (batch_id + 1) * self.batch_size[key]
		if start >= len(self.index[key]):
			return None
		if ignore_left_samples and end > len(self.index[key]):
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
		return list(map(lambda word: self.word2id.get(word, self.unk_id), sen))

	def trim_index(self, index):
		'''Trim index. There will be two steps:
				* find first `<eos>` and abondon words after it (included the `<eos>`).
				* ignore `<pad>` s at the end of the sentence.

		Arguments:
				index (list): a list of int

		Examples:
				>>> dataloader.index_to_sen(
				...		[])
				>>>

		Todo:
				* fix the missing example
		'''

		index = trim_before_target(list(index), self.eos_id)
		idx = len(index)
		while index[idx - 1] == self.pad_id:
			idx -= 1
		index = index[:idx]
		return index

	def index_to_sen(self, index, trim=True):
		'''Convert a sentences from index to string representation

		Arguments:
				index (list): a list of int
				trim (bool): if True, call :func:`trim_index` before convertion.

		Examples:
				>>> dataloader.index_to_sen(
				...		[])
				>>>

		Todo:
				* fix the missing example
		'''
		if trim:
			index = self.trim_index(index)
		return list(map(lambda word: self.vocab_list[word], index))

	def get_teacher_forcing_metric(self, gen_prob_key="gen_prob"):
		'''Get metric for teacher-forcing mode.

		It contains:

		* :class:`.metric.PerlplexityMetric`

		Arguments:
				gen_prob_key (str): default: `gen_prob`. Refer to :class:`.metric.PerlplexityMetric`
		'''
		return PerlplexityMetric(self,
								 data_key='sentence',
								 data_len_key='sentence_length',
								 gen_prob_key=gen_prob_key)

	def get_inference_metric(self, gen_key="gen"):
		'''Get metric for inference.

		It contains:

		* :class:`.metric.BleuCorpusMetric`
		# * :class:`.metric.SingleDialogRecorder`

		Arguments:
				gen_key (str): default: "gen". Refer to :class:`.metric.BleuCorpusMetric` or
							   :class:`.metric.SingleDialogRecorder`
		'''
		metric = MetricChain()
		metric.add_metric(BleuCorpusMetric(self, data_key='sentence', gen_key=gen_key))
		metric.add_metric(LanguageGenerationRecorder(self,
													 sentence_key="sentence",
													 gen_key=gen_key))
		return metric


class MSCOCO(LanguageGeneration):
	'''A dataloder for preprocessed MSCOCO dataset.
	source: http://images.cocodataset.org/annotations/annotations_trainval2017.zip)

	Arguments:
			file_path (str): a str indicates the dir of MSCOCO dataset.
			min_vocab_times (int): A cut-off threshold of `UNK` tokens. All tokens appear
					less than `min_vocab_times`	will be replaced by `<unk>`. Default: 10.
			max_sen_length (int): All sentences longer than `max_sen_length` will be shortened
					to first `max_sen_length` tokens. Default: 50.

	Refer to :class:`.LanguageGeneration` for attributes.

	Todo:
			* add references
	'''

	def __init__(self, file_path, min_vocab_times=10, max_sen_length=50):
		self._file_path = file_path
		self._min_vocab_times = min_vocab_times
		self._max_sen_length = max_sen_length
		super(MSCOCO, self).__init__()

	def _load_data(self):
		r'''Loading dataset, invoked by LanguageGeneration.__init__
		'''
		origin_data = {}
		for key in self.key_name:
			f_file = open("%s/mscoco_%s.txt" % (self._file_path, key))
			origin_data[key] = list(
				map(lambda line: line.split(), f_file.readlines()))

		vocab = list(chain(*(origin_data['train'])))
		# Important: Sort the words preventing the index changes between
		# different runs
		vocab = sorted(Counter(vocab).most_common(),
					   key=lambda pair: (-pair[1], pair[0]))
		left_vocab = list(
			filter(
				lambda x: x[1] >= self._min_vocab_times,
				vocab))
		vocab_list = self.ext_vocab + list(map(lambda x: x[0], left_vocab))
		word2id = {w: i for i, w in enumerate(vocab_list)}
		print("vocab list length = %d" % len(vocab_list))

		def line2id(line):
			return ([self.go_id] +
					list(map(lambda word: word2id[word] if word in word2id else self.unk_id, line))
					+ [self.eos_id])[:self._max_sen_length]

		data = {}
		for key in self.key_name:
			data[key] = list(map(line2id, origin_data[key]))

			vocab = list(chain(*(origin_data[key])))
			vocab_num = len(vocab)
			oov_num = len(
				list(
					filter(
						lambda word: word not in word2id,
						vocab)))
			length = list(
				map(len, origin_data[key]))
			cut_num = np.sum(
				np.maximum(
					np.array(length) -
					self._max_sen_length +
					1,
					0))
			print(
				"%s set. OOV rate: %f, max length before cut: %d, cut word rate: %f" %
				(key, oov_num / vocab_num, max(length), cut_num / vocab_num))
		return vocab_list, data
