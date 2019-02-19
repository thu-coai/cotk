"""Dataloader for language generation"""
from collections import Counter
from itertools import chain

import numpy as np

from .dataloader import BasicLanguageGeneration
from ..metric import MetricChain, PerlplexityMetric, LanguageGenerationRecorder

# pylint: disable=W0223
class LanguageGeneration(BasicLanguageGeneration):
	r"""Base class for language modelling datasets. This is an abstract class.

	Arguments:{ARGUMENTS}

	Attributes:{ATTRIBUTES}
	"""

	ARGUMENTS = BasicLanguageGeneration.ARGUMENTS
	ATTRIBUTES = BasicLanguageGeneration.ATTRIBUTES

	def get_batch(self, key, index):
		'''Get a batch of specified `index`.

		Arguments:
			key (str): must be contained in `key_name`
			index (list): a list of specified index

		Returns:
			(dict): A dict at least contains:

				* sentence_length(list): A 1-d list, the length of sentence in each batch.
				  Size: `[batch_size]`
				* sentence(:class:`numpy.array`): A 2-d padding array containing id of words.
				  Only provide valid words. `unk_id` will be used if a word is not valid.
				  Size: `[batch_size, max(sent_length)]`
				* sentence_allwords(:class:`numpy.array`): A 2-d padding array containing id of words.
				  Provide both valid and invalid words.
				  Size: `[batch_size, max(sent_length)]`

		Examples:
			>>> # vocab_list = ["<pad>", "<unk>", "<go>", "<eos>", "how", "are", "you",
			>>> #	"hello", "i", "am", "fine"]
			>>> dataloader.get_batch('train', [0, 1])
			{
				"sentence": [
						[2, 4, 5, 6, 3],   # first sentence: <go> how are you <eos>
						[2, 7, 3, 0, 0],   # second sentence:  <go> hello <eos> <pad> <pad>
					],
					"sentence_length": [5, 3], # length of sentences
			}

		Todo:
			* add invalid_vocab examples
			* mark which array is np.array
		'''
		if key not in self.key_name:
			raise ValueError("No set named %s." % key)
		res = {}
		batch_size = len(index)
		res["sentence_length"] = np.array( \
			list(map(lambda i: len(self.data[key]['sen'][i]), index)))
		res_sent = res["sentence"] = np.zeros( \
			(batch_size, np.max(res["sentence_length"])), dtype=int)
		for i, j in enumerate(index):
			sentence = self.data[key]['sen'][j]
			res["sentence"][i, :len(sentence)] = sentence

		res["sentence_allwords"] = res_sent.copy()
		res_sent[res_sent >= self.valid_vocab_len] = self.unk_id
		return res

	def get_teacher_forcing_metric(self, gen_prob_key="gen_prob"):
		'''Get metric for teacher-forcing mode.

		It contains:

		* :class:`.metric.PerlplexityMetric`

		Arguments:
				gen_prob_key (str): default: `gen_prob`. Refer to :class:`.metric.PerlplexityMetric`
		'''
		return PerlplexityMetric(self, \
								 reference_key='sentence', \
								 reference_len_key='sentence_length', \
								 gen_prob_key=gen_prob_key)

	def get_inference_metric(self, gen_key="gen"):
		'''Get metric for inference.

		It contains:

		* :class:`.metric.LanguageGenerationRecorder`

		Arguments:
				gen_key (str): default: "gen". Refer to :class:`.metric.LanguageGenerationRecorder`
		'''
		metric = MetricChain()
		metric.add_metric(LanguageGenerationRecorder(self, \
													 gen_key=gen_key))
		return metric


class MSCOCO(LanguageGeneration):
	'''A dataloder for preprocessed MSCOCO dataset.

	Arguments:
			file_path (str): a str indicates the dir of MSCOCO dataset.
			valid_vocab_times (int): A cut-off threshold of valid tokens. All tokens appear
					not less than `min_vocab_times` in **training set** will be marked as valid words.
					Default: 10.
			max_sen_length (int): All sentences longer than `max_sen_length` will be shortened
					to first `max_sen_length` tokens. Default: 50.
			invalid_vocab_times (int):  A cut-off threshold of invalid tokens. All tokens appear
					not less than `invalid_vocab_times` in the **whole dataset** (except valid words) will be
					marked as invalid words. Otherwise, they are unknown words, both in training or
					testing stages. Default: 0 (No unknown words).

	Refer to :class:`.LanguageGeneration` for attributes and methods.

	References:
		[1] http://images.cocodataset.org/annotations/annotations_trainval2017.zip

		[2] Lin T Y, Maire M, Belongie S, et al. Microsoft COCO: Common Objects in Context. ECCV 2014.

	'''

	def __init__(self, file_path, min_vocab_times=10, max_sen_length=50, invalid_vocab_times=0):
		self._file_path = file_path
		self._min_vocab_times = min_vocab_times
		self._max_sen_length = max_sen_length
		self._invalid_vocab_times = invalid_vocab_times
		super(MSCOCO, self).__init__()

	def _load_data(self):
		r'''Loading dataset, invoked by `LanguageGeneration.__init__`
		'''
		origin_data = {}
		for key in self.key_name:
			f_file = open("%s/mscoco_%s.txt" % (self._file_path, key))
			origin_data[key] = {}
			origin_data[key]['sen'] = list( \
				map(lambda line: line.split(), f_file.readlines()))

		raw_vocab_list = list(chain(*(origin_data['train']['sen'])))
		# Important: Sort the words preventing the index changes between
		# different runs
		vocab = sorted(Counter(raw_vocab_list).most_common(), \
					   key=lambda pair: (-pair[1], pair[0]))
		left_vocab = list( \
			filter( \
				lambda x: x[1] >= self._min_vocab_times, \
				vocab))
		vocab_list = self.ext_vocab + list(map(lambda x: x[0], left_vocab))
		valid_vocab_len = len(vocab_list)
		valid_vocab_set = set(vocab_list)

		for key in self.key_name:
			if key == 'train':
				continue
			raw_vocab_list.extend(list(chain(*(origin_data[key]['sen']))))
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
		def line2id(line):
			return ([self.go_id] + \
					list(map(lambda word: word2id[word] if word in word2id else self.unk_id, line)) \
					+ [self.eos_id])[:self._max_sen_length]

		data = {}
		data_size = {}
		for key in self.key_name:
			data[key] = {}
			data[key]['sen'] = list(map(line2id, origin_data[key]['sen']))
			data_size[key] = len(data[key]['sen'])

			vocab = list(chain(*(origin_data[key]['sen'])))
			vocab_num = len(vocab)
			oov_num = len( \
				list( \
					filter( \
						lambda word: word not in word2id, \
						vocab)))
			invalid_num = len( \
				list( \
					filter( \
						lambda word: word not in valid_vocab_set, \
						vocab))) - oov_num
			length = list( \
				map(len, origin_data[key]['sen']))
			cut_num = np.sum( \
				np.maximum( \
					np.array(length) - \
					self._max_sen_length + \
					1, \
					0))
			print( \
				"%s set. invalid rate: %f, unknown rate: %f, max length before cut: %d, cut word rate: %f" % \
				(key, invalid_num / vocab_num, oov_num / vocab_num, max(length), cut_num / vocab_num))
		return vocab_list, valid_vocab_len, data, data_size
