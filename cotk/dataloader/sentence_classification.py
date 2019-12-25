"""Dataloader for language generation"""
from collections import Counter
from itertools import chain
import os
import json

import numpy as np

# from .._utils.unordered_hash import UnorderedSha256
from .._utils.file_utils import get_resource_file_path
from .._utils import hooks
from .dataloader import LanguageProcessingBase
from ..metric import MetricChain, AccuracyMetric


# pylint: disable=W0223
class SentenceClassification(LanguageProcessingBase):
	r"""Base class for sentence classification datasets. This is an abstract class.

	Arguments:{ARGUMENTS}

	Attributes:{ATTRIBUTES}
	"""

	_version = 1

	ARGUMENTS = LanguageProcessingBase.ARGUMENTS
	ATTRIBUTES = LanguageProcessingBase.ATTRIBUTES

	def get_batch(self, key, indexes):
		'''Get a batch of specified `indexes`.

		Arguments:
			key (str): must be contained in `key_name`
			indexes (list): a list of specified indexes

		Returns:
			(dict): A dict at least contains:

				* sent_length(:class:`numpy.array`): A 1-d array, the length of sentence in each batch.
				  Size: `[batch_size]`
				* sent(:class:`numpy.array`): A 2-d padding array containing id of words.
				  Only provide valid words. `unk_id` will be used if a word is not valid.
				  Size: `[batch_size, max(sent_length)]`
				* label(:class:`numpy.array`): A 1-d array, the label of sentence in each batch.
				* sent_allvocabs(:class:`numpy.array`): A 2-d padding array containing id of words.
				  Provide both valid and invalid words.
				  Size: `[batch_size, max(sent_length)]`

		Examples:
			>>> # all_vocab_list = ["<pad>", "<unk>", "<go>", "<eos>", "how", "are", "you",
			>>> #	"hello", "i", "am", "fine"]
			>>> # vocab_size = 9
			>>> # vocab_list = ["<pad>", "<unk>", "<go>", "<eos>", "how", "are", "you", "hello", "i"]
			>>> dataloader.get_batch('train', [0, 1, 2])
			{
				"sent": numpy.array([
						[2, 4, 5, 6, 3, 0],   # first sentence: <go> how are you <eos> <pad>
						[2, 7, 3, 0, 0, 0],   # second sentence:  <go> hello <eos> <pad> <pad> <pad>
						[2, 7, 8, 1, 1, 3]    # third sentence: <go> hello i <unk> <unk> <eos>
					]),
				"label": numpy.array([1, 2, 0]) # label of sentences
				"sent_length": numpy.array([5, 3, 6]), # length of sentences
				"sent_allvocabs": numpy.array([
						[2, 4, 5, 6, 3, 0],   # first sentence: <go> how are you <eos> <pad>
						[2, 7, 3, 0, 0, 0],   # second sentence:  <go> hello <eos> <pad> <pad> <pad>
						[2, 7, 8, 9, 10, 3]   # third sentence: <go> hello i am fine <eos>
					]),
			}
		'''
		if key not in self.key_name:
			raise ValueError("No set named %s." % key)
		res = {}
		batch_size = len(indexes)
		res["sent_length"] = np.array( \
			list(map(lambda i: len(self.data[key]['sent'][i]), indexes)), dtype=int)
		res_sent = res["sent"] = np.zeros( \
			(batch_size, np.max(res["sent_length"])), dtype=int)
		res["label"] = np.zeros(batch_size, dtype=int)
		for i, j in enumerate(indexes):
			sentence = self.data[key]['sent'][j]
			res["sent"][i, :len(sentence)] = sentence
			res["label"][i] = self.data[key]['label'][j]

		res["sent_allvocabs"] = res_sent.copy()
		res_sent[res_sent >= self.valid_vocab_len] = self.unk_id
		return res

	def get_metric(self, prediction_key="prediction"):
		'''Get metrics for accuracy. In other words, this function
		provides metrics for sentence classification task.

		It contains:

			* :class:`.metric.AccuracyMetric`

		Arguments:
			prediction_key (str): The key of prediction over sentences.
				Refer to :class:`.metric.AccuracyMetric`. Default: ``prediction``.

		Returns:
			A :class:`.metric.MetricChain` object.
		'''
		metric = MetricChain()
		metric.add_metric(AccuracyMetric(self, \
										 label_key='label', \
										 prediction_key=prediction_key))
		return metric


class SST(SentenceClassification):
	'''A dataloader for preprocessed SST dataset.

	Arguments:
			file_id (str): a str indicates the source of SST dataset.
			file_type (str): a str indicates the type of SST dataset. Default: "SST"
			valid_vocab_times (int): A cut-off threshold of valid tokens. All tokens appear
					not less than `min_vocab_times` in **training set** will be marked as valid words.
					Default: 10.
			max_sent_length (int): All sentences longer than `max_sent_length` will be shortened
					to first `max_sent_length` tokens. Default: 50.
			invalid_vocab_times (int):  A cut-off threshold of invalid tokens. All tokens appear
					not less than `invalid_vocab_times` in the **whole dataset** (except valid words) will be
					marked as invalid words. Otherwise, they are unknown words, both in training or
					testing stages. Default: 0 (No unknown words).

	Refer to :class:`.SentenceClassification` for attributes and methods.

	References:
		[1] http://images.cocodataset.org/annotations/annotations_trainval2017.zip

		[2] Lin T Y, Maire M, Belongie S, et al. Microsoft COCO: Common Objects in Context. ECCV 2014.

	'''

	@hooks.hook_dataloader
	def __init__(self, file_id, min_vocab_times=10, \
				 max_sent_length=50, invalid_vocab_times=0):
		self._file_id = file_id
		self._file_path = get_resource_file_path(file_id)
		self._min_vocab_times = min_vocab_times
		self._max_sent_length = max_sent_length
		self._invalid_vocab_times = invalid_vocab_times
		super(SST, self).__init__()

	def _load_data(self):
		r'''Loading dataset, invoked by `LanguageProcessingBase.__init__`
		'''
		vocab_list, valid_vocab_len, data, data_size = \
			super()._general_load_data(self._file_path,
									[['sent', 'Sentence']],
									self._min_vocab_times,
									self._max_sent_length,
									None,
									self._invalid_vocab_times)
		for key in self.key_name:
			with open(os.path.join(self._file_path, key + '_labels.json'), 'r', encoding='utf-8') as fp:
				data[key]['label'] = json.load(fp)
		return vocab_list, valid_vocab_len, data, data_size

	def tokenize(self, sentence):
		r'''Convert sentence(str) to list of token(str)

		Arguments:
			sentence (str)

		Returns:
			sent (list): list of token(str)
		'''
		# return [x.split(' ')[-1].lower() for x in sentence if x != '']
		return super().tokenize(sentence, True, 'space')
