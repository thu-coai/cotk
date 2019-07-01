'''
A module for single turn dialog.
'''
from collections import Counter
from itertools import chain

import numpy as np

from .._utils.unordered_hash import UnorderedSha256
from .._utils.file_utils import get_resource_file_path
from .dataloader import GenerationBase
from ..metric import MetricChain, PerplexityMetric, BleuCorpusMetric, SingleTurnDialogRecorder, \
			HashValueRecorder

# pylint: disable=W0223
class SingleTurnDialog(GenerationBase):
	r"""Base class for single-turn dialog datasets. This is an abstract class.

	Arguments:{ARGUMENTS}

	Attributes:{ATTRIBUTES}
	"""

	ARGUMENTS = GenerationBase.ARGUMENTS
	ATTRIBUTES = GenerationBase.ATTRIBUTES

	def get_batch(self, key, index, needhash=False):
		'''Get a batch of specified `index`.
		Arguments:
			key (str): must be contained in `key_name`
			index (list): a list of specified index
			needhash (bool): whether to return a hashvalue \
				representing this batch of data. Default: False.

		Returns:
			(dict): A dict at least contains:

			* post_length (list): A 1-d list, the length of post in each batch.
			  Size: `[batch_size]`
			* post (:class:`numpy.array`): A 2-d padding array containing id of words in posts.
			  Only provide valid words. `unk_id` will be used if a word is not valid.
			  Size: `[batch_size, max(sent_length)]`
			* post_allvocabs (:class:`numpy.array`): A 2-d padding array containing id of words in posts.
			  Provide both valid and invalid vocabs.
			  Size: `[batch_size, max(sent_length)]`
			* resp_length (list): A 1-d list, the length of response in each batch.
			  Size: `[batch_size]`
			* resp (:class:`numpy.array`): A 2-d padding array containing id of words in responses.
			  Only provide valid vocabs. `unk_id` will be used if a word is not valid.
			  Size: `[batch_size, max(sent_length)]`
			* resp_allvocabs (:class:`numpy.array`): A 2-d padding array containing id of words in responses.
			  Provide both valid and invalid vocabs.
			  Size: `[batch_size, max(sent_length)]`
			* hashvalue(bytes): (If `needhash` is True.) A bytes representing hash value of the data.

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

		Todo:
			* add invalid_vocab examples
			* mark which is np.array
		'''
		if key not in self.key_name:
			raise ValueError("No set named %s." % key)
		res = {}
		batch_size = len(index)
		res["post_length"] = np.array(list(map(lambda i: len(self.data[key]['post'][i]), index)))
		res["resp_length"] = np.array(list(map(lambda i: len(self.data[key]['resp'][i]), index)))
		res_post = res["post"] = np.zeros((batch_size, np.max(res["post_length"])), dtype=int)
		res_resp = res["resp"] = np.zeros((batch_size, np.max(res["resp_length"])), dtype=int)
		for i, j in enumerate(index):
			post = self.data[key]['post'][j]
			resp = self.data[key]['resp'][j]
			res_post[i, :len(post)] = post
			res_resp[i, :len(resp)] = resp

		if needhash:
			unordered_hash = UnorderedSha256()
			for j in index:
				unordered_hash.update_data(repr((\
					self.data[key]['post'][j], \
					self.data[key]['resp'][j], \
					self.valid_vocab_len)).encode())
			res["hashvalue"] = unordered_hash.digest()

		res["post_allvocabs"] = res_post.copy()
		res["resp_allvocabs"] = res_resp.copy()
		res_post[res_post >= self.valid_vocab_len] = self.unk_id
		res_resp[res_resp >= self.valid_vocab_len] = self.unk_id
		return res

	def get_teacher_forcing_metric(self, gen_log_prob_key="gen_log_prob"):
		'''Get metric for teacher-forcing mode.

		It contains:

		* :class:`.metric.PerplexityMetric`

		Arguments:
			gen_prob_key (str): default: `gen_prob`. Refer to :class:`.metric.PerplexityMetric`
		'''
		metric = MetricChain()
		metric.add_metric(HashValueRecorder(hash_key="teacher_forcing_hashvalue"))
		metric.add_metric(PerplexityMetric(self, gen_log_prob_key=gen_log_prob_key))
		return metric

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
		metric.add_metric(HashValueRecorder(hash_key="inference_hashvalue"))
		metric.add_metric(BleuCorpusMetric(self, gen_key=gen_key))
		metric.add_metric(SingleTurnDialogRecorder(self, gen_key=gen_key))
		return metric

class OpenSubtitles(SingleTurnDialog):
	'''A dataloader for OpenSubtitles dataset.

	Arguments:
		file_id (str): a str indicates the source of OpenSubtitles dataset.
		file_type (str): a str indicates the type of OpenSubtitles dataset. Default: "OpenSubtitles"
		min_vocab_times (int): A cut-off threshold of `UNK` tokens. All tokens appear
			less than `min_vocab_times`	will be replaced by `<unk>`. Default: 10.
		max_sen_length (int): All sentences longer than `max_sen_length` will be shortened
			to first `max_sen_length` tokens. Default: 50.
		invalid_vocab_times (int):  A cut-off threshold of invalid tokens. All tokens appear
			not less than `invalid_vocab_times` in the **whole dataset** (except valid words) will be
			marked as invalid vocabs. Otherwise, they are unknown words, both in training or
			testing stages. Default: 0 (No unknown words).

	Refer to :class:`.SingleTurnDialog` for attributes and methods.

	References:
		[1] http://opus.nlpl.eu/OpenSubtitles.php

		[2] P. Lison and J. Tiedemann, OpenSubtitles2016: Extracting Large Parallel Corpora from
		Movie and TV Subtitles.(LREC 2016)
	'''
	def __init__(self, file_id, min_vocab_times=10, \
			max_sen_length=50, invalid_vocab_times=0):
		self._file_id = file_id
		self._file_path = get_resource_file_path(file_id)
		self._min_vocab_times = min_vocab_times
		self._max_sen_length = max_sen_length
		self._invalid_vocab_times = invalid_vocab_times
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

		raw_vocab_list = list(chain(*(origin_data['train']['post'] + origin_data['train']['resp'])))
		# Important: Sort the words preventing the index changes between different runs
		vocab = sorted(Counter(raw_vocab_list).most_common(), key=lambda pair: (-pair[1], pair[0]))
		left_vocab = list(filter(lambda x: x[1] >= self._min_vocab_times, vocab))
		vocab_list = self.ext_vocab + list(map(lambda x: x[0], left_vocab))
		valid_vocab_len = len(vocab_list)
		valid_vocab_set = set(vocab_list)

		for key in self.key_name:
			if key == 'train':
				continue
			raw_vocab_list.extend(list(chain(*(origin_data[key]['post'] + origin_data[key]['resp']))))
		vocab = sorted(Counter(raw_vocab_list).most_common(), \
					   key=lambda pair: (-pair[1], pair[0]))
		left_vocab = list( \
			filter( \
				lambda x: x[1] >= self._invalid_vocab_times and x[0] not in valid_vocab_set, \
				vocab))
		vocab_list.extend(list(map(lambda x: x[0], left_vocab)))

		print("valid vocab list length = %d" % valid_vocab_len)
		print("vocab list length = %d" % len(vocab_list))

		word2id = {w: i for i, w in enumerate(vocab_list)}
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
			invalid_num = len( \
				list( \
					filter( \
						lambda word: word not in valid_vocab_set, \
						vocab))) - oov_num
			length = list(map(len, origin_data[key]['post'] + origin_data[key]['resp']))
			cut_num = np.sum(np.maximum(np.array(length) - self._max_sen_length + 1, 0))
			print("%s set. invalid rate: %f, unknown rate: %f, max length before cut: %d, \
					cut word rate: %f" % \
					(key, invalid_num / vocab_num, oov_num / vocab_num, max(length), cut_num / vocab_num))
		return vocab_list, valid_vocab_len, data, data_size
