'''
A module for single turn dialog.
'''
from collections import Counter
from itertools import chain

import numpy as np

from .dataloader import BasicLanguageGeneration
from ..metric import MetricChain, PerlplexityMetric, BleuCorpusMetric, SingleTurnDialogRecorder

# pylint: disable=W0223
class SingleTurnDialog(BasicLanguageGeneration):
	r"""Base class for single-turn dialog datasets. This is an abstract class.

	Arguments:
			end_token (int): the special token that stands for end. default: `3("<eos>")`
			ext_vocab (list): special tokens. default: `["<pad>", "<unk>", "<go>", "<eos>"]`
			key_name (list): name of subsets of the data. default: `["train", "dev", "test"]`

	Attributes:
			ext_vocab (list): special tokens, be placed at beginning of `vocab_list`.
					For example: `["<pad>", "<unk>", "<go>", "<eos>"]`
			pad_id (int): token for padding, always equal to `0`
			unk_id (int): token for unknown words, always equal to `1`
			go_id (int): token at the beginning of sentences, always equal to `2`
			eos_id (int): token at the end of sentences, always equal to `3`
			key_name (list): name of subsets of the data. For example: `["train", "dev", "test"]`
			vocab_list (list): vocabulary list of the datasets.
			word2id (dict): a dict mapping tokens to index.
					Maybe you want to use :meth:`sen_to_index` instead.
			end_token (int): token for end. default: equals to `eos_id`
	"""

	def get_batch(self, key, index):
		'''Get a batch of specified `index`.
		Arguments:
			key (str): must be contained in `key_name`
			index (list): a list of specified index

		Returns:
			A dict at least contains ``post``, ``post_length``, ``resp``,
			``resp_length``. See the example belows.

		Examples:
			>>> # vocab_list = ["<pad>", "<unk>", "<go>", "<eos>", "how", "are", "you",
			>>> #	"hello", "i", "am", "fine"]
			>>> dataloader.get_batch('train', [0, 1])
			{
				"post": [
					[2, 4, 5, 6, 3],   # first post: <go> how are you <eos>
					[2, 7, 3, 0, 0],   # second post: <go> hello <eos> <pad> <pad>
				],
				"resp": [
					[2, 8, 9, 10, 3],  # first response: <go> i am fine <eos>
					[2, 7, 3, 0, 0],   # second response: <go> hello <eos> <pad> <pad>
				],
				"post_length": [5, 3], # length of posts
				"resp_length": [5, 3], # length of responses
			}
		'''
		if key not in self.key_name:
			raise ValueError("No set named %s." % key)
		res = {}
		batch_size = len(index)
		res["post_length"] = np.array(list(map(lambda i: len(self.data[key]['post'][i]), index)))
		res["resp_length"] = np.array(list(map(lambda i: len(self.data[key]['resp'][i]), index)))
		res["post"] = np.zeros((batch_size, np.max(res["post_length"])), dtype=int)
		res["resp"] = np.zeros((batch_size, np.max(res["resp_length"])), dtype=int)
		for i, j in enumerate(index):
			post = self.data[key]['post'][j]
			resp = self.data[key]['resp'][j]
			res["post"][i, :len(post)] = post
			res["resp"][i, :len(resp)] = resp
		return res

	def get_teacher_forcing_metric(self, gen_prob_key="gen_prob"):
		'''Get metric for teacher-forcing mode.

		It contains:

		* :class:`.metric.PerlplexityMetric`

		Arguments:
			gen_prob_key (str): default: `gen_prob`. Refer to :class:`.metric.PerlplexityMetric`
		'''
		return PerlplexityMetric(self, gen_prob_key=gen_prob_key)

	def get_inference_metric(self, gen_key="gen"):
		'''Get metric for inference.

		It contains:

		* :class:`.metric.BleuCorpusMetric`
		* :class:`.metric.SingleDialogRecorder`

		Arguments:
			gen_key (str): default: "gen". Refer to :class:`.metric.BleuCorpusMetric` or
			               :class:`.metric.SingleDialogRecorder`
		'''
		metric = MetricChain()
		metric.add_metric(BleuCorpusMetric(self, gen_key=gen_key))
		metric.add_metric(SingleTurnDialogRecorder(self, gen_key=gen_key))
		return metric

class OpenSubtitles(SingleTurnDialog):
	'''A dataloder for OpenSubtitles dataset.

	Arguments:
		file_path (str): a str indicates the dir of OpenSubtitles dataset.
		min_vocab_times (int): A cut-off threshold of `UNK` tokens. All tokens appear
			less than `min_vocab_times`	will be replaced by `<unk>`. Default: 10.
		max_sen_length (int): All sentences longer than `max_sen_length` will be shortened
			to first `max_sen_length` tokens. Default: 50.

	Refer to :class:`.SingleTurnDialog` for attributes and methods.

	References:
		[1] http://opus.nlpl.eu/OpenSubtitles.php

		[2] P. Lison and J. Tiedemann, OpenSubtitles2016: Extracting Large Parallel Corpora from
		Movie and TV Subtitles.(LREC 2016)
	'''
	def __init__(self, file_path, min_vocab_times=10, max_sen_length=50):
		self._file_path = file_path
		self._min_vocab_times = min_vocab_times
		self._max_sen_length = max_sen_length
		super(OpenSubtitles, self).__init__()

	def _load_data(self):
		r'''Loading dataset, invoked by `SingleTurnDialog.__init__`
		'''
		origin_data = {}
		for key in self.key_name:
			f_file = open("%s/opensub_pair_%s.post" % (self._file_path, key))
			g_file = open("%s/opensub_pair_%s.response" % (self._file_path, key))
			origin_data[key] = {}
			origin_data[key]['post'] = list(map(lambda line: line.split(), f_file.readlines()))
			origin_data[key]['resp'] = list(map(lambda line: line.split(), g_file.readlines()))

		vocab = list(chain(*(origin_data['train']['post'] + origin_data['train']['resp'])))
		# Important: Sort the words preventing the index changes between different runs
		vocab = sorted(Counter(vocab).most_common(), key=lambda pair: (-pair[1], pair[0]))
		left_vocab = list(filter(lambda x: x[1] >= self._min_vocab_times, vocab))
		vocab_list = self.ext_vocab + list(map(lambda x: x[0], left_vocab))
		word2id = {w: i for i, w in enumerate(vocab_list)}
		print("vocab list length = %d" % len(vocab_list))

		line2id = lambda line: ([self.go_id] + \
					list(map(lambda word: word2id[word] if word in word2id else self.unk_id, line)) + \
					[self.eos_id])[:self._max_sen_length]

		data = {}
		data_size = {}
		for key in self.key_name:
			data[key] = {}

			data[key]['post'] = list(map(line2id, origin_data[key]['post']))
			data[key]['resp'] = list(map(line2id, origin_data[key]['resp']))
			data_size[key] = len(data[key]['post'])
			vocab = list(chain(*(origin_data[key]['post'] + origin_data[key]['resp'])))
			vocab_num = len(vocab)
			oov_num = len(list(filter(lambda word: word not in word2id, vocab)))
			length = list(map(len, origin_data[key]['post'] + origin_data[key]['resp']))
			cut_num = np.sum(np.maximum(np.array(length) - self._max_sen_length + 1, 0))
			print("%s set. OOV rate: %f, max length before cut: %d, cut word rate: %f" % \
					(key, oov_num / vocab_num, max(length), cut_num / vocab_num))
		return vocab_list, data, data_size
