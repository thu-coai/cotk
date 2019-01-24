"""
A module for multi turn dialog.
"""
import csv
from collections import Counter
from itertools import chain

import numpy as np
from .dataloader import BasicLanguageGeneration
from ..metric import MetricChain, MultiTurnPerplexityMetric, MultiTurnBleuCorpusMetric, \
	MultiTurnDialogRecorder

# pylint: disable=W0223
class MultiTurnDialog(BasicLanguageGeneration):
	r"""Base class for multi-turn dialog datasets. This is an abstract class.

	Arguments:
			end_token (int): the special token that stands for end. default: `4("<eot>")`
			ext_vocab (list): special tokens. default: `["<pad>", "<unk>", "<go>", "<eos>", "<eot>"]`
			key_name (list): name of subsets of the data. default: `["train", "dev", "test"]`

	Attributes:
			ext_vocab (list): special tokens, be placed at beginning of `vocab_list`.
					For example: `["<pad>", "<unk>", "<go>", "<eos>", "<eot>"]`
			pad_id (int): token for padding, always equal to `0`
			unk_id (int): token for unknown words, always equal to `1`
			go_id (int): token at the beginning of sentences, always equal to `2`
			eos_id (int): token at the end of sentences, always equal to `3`
			eot_id (int): token at the end of turns, always equal to `4`
			key_name (list): name of subsets of the data. For example: `["train", "dev", "test"]`
			vocab_list (list): vocabulary list of the datasets.
			word2id (dict): a dict mapping tokens to index.
					Maybe you want to use :meth:`sen_to_index` instead.
			end_token (int): token for end. default: equals to `eot_id`
	"""
	def __init__(self, \
				 end_token=None, \
				 ext_vocab=None, \
				 key_name=None,	\
		):
		ext_vocab = ext_vocab or ["<pad>", "<unk>", "<go>", "<eos>", "<eot>"]
		self.eot_id = ext_vocab.index("<eot>")
		super().__init__(end_token or self.eot_id, ext_vocab, key_name)

	def get_batch(self, key, index):
		'''Get a batch of specified `index`.

		Arguments:
			key (str): must be contained in `key_name`
			index (list): a list of specified index

		Returns:
			(dict): A dict at least contains:

				* turn_length(list): A 1-d list, the number of turns in sessions. \
					Size: `[batch_size]`
				* sent_length(list): A 2-d non-padded list, the length of sentence in turns. \
					The second dimension is various in different session. \
					Length of outer list: `[batch_size]`
				* sent(:class:numpy.array): A 3-d padding array containing id of words. \
					Size: `[batch_size, max(turn_length[i]), max(sent_length)]`

			See the example belows.

		Examples:
			>>> dataloader.get_batch('train', 1)
			>>>

		Todo:
			* fix the missing example
		'''
		if key not in self.key_name:
			raise ValueError("No set named %s." % key)
		res = {}
		res["turn_length"] = [len(self.data[key]['session'][i]) for i in index]
		res["sent_length"] = []
		for i in index:
			sent_length = [len(sent) for sent in self.data[key]['session'][i]]
			res["sent_length"].append(sent_length)
		res["sent"] = np.zeros((len(index), np.max(res['turn_length']), \
			np.max(list(chain(*res['sent_length'])))), dtype=int)
		for i, index_i in enumerate(index):
			for j, sent in enumerate(self.data[key]['session'][index_i]):
				res['sent'][i, j, :len(sent)] = sent
		return res

	def multi_turn_trim_index(self, index):
		'''Trim indexes for multi turn dialog. There will be 3 steps:
			* For every turn, if there is an `<eot>`, \
				find first `<eot>` and abondon words after it (included the `<eot>`).
			* If the sentence after triming is empty, discard this turn and the turn after it.
			* Ignore `<pad>` s at the end of every turn.

		Arguments:
			index (list or :class:numpy.array): a 2-d array of int.
			Size: [turn_length, max_sent_length]

		Examples:

		Todo:
			* fix the missing example
		'''

		res = []
		for turn_index in index:
			turn_trim = self.trim_index(turn_index)
			if turn_trim:
				res.append(turn_trim)
			else:
				break
		return res

	def multi_turn_sen_to_index(self, session):
		'''Convert a session from string to index representation.

		Arguments:
			sen (list): a 2-d list of str, representing each token of the session.

		Examples:

		Todo:
			* fix the missing example
		'''
		return list(map(lambda sent: list(map( \
			lambda word: self.word2id.get(word, self.unk_id), sent)), \
		session))

	def multi_turn_index_to_sen(self, index, trim=True):
		'''Convert a session from index to string representation

		Arguments:
			index (list or :class:numpy.array): a 2-d array of int.
			Size: [turn_length, max_sent_length]
			trim (bool): if True, call :func:`multi_turn_trim_index` before convertion.

		Examples:

		Todo:
			* fix the missing example
		'''
		if trim:
			index = self.multi_turn_trim_index(index)
		return list(map(lambda sent: \
			list(map(lambda word: self.vocab_list[word], sent)), \
			index))

	def get_teacher_forcing_metric(self, gen_prob_key="gen_prob"):
		'''Get metric for teacher-forcing mode.

		It contains:

		* :class:`.metric.MultiTurnPerplexityMetric`

		Arguments:
			gen_prob_key (str): default: `gen_prob`. Refer to :class:`.metric.PerlplexityMetric`
		'''
		return MultiTurnPerplexityMetric(self, gen_prob_key=gen_prob_key)

	def get_inference_metric(self, gen_key="gen"):
		'''Get metric for inference.

		It contains:

		* :class:`.metric.BleuCorpusMetric`
		* :class:`.metric.MultiTurnDialogRecorder`

		Arguments:
			gen_key (str): default: "gen". Refer to :class:`.metric.BleuCorpusMetric` or
					   :class:`.metric.MultiTurnDialogRecorder`
		'''
		metric = MetricChain()
		metric.add_metric(MultiTurnBleuCorpusMetric(self, gen_key=gen_key))
		metric.add_metric(MultiTurnDialogRecorder(self, gen_key=gen_key))
		return metric

class UbuntuCorpus(MultiTurnDialog):
	'''A dataloder for OpenSubtitles dataset.

	Arguments:
		file_path (str): a str indicates the dir of OpenSubtitles dataset.
		min_vocab_times (int): A cut-off threshold of `UNK` tokens. All tokens appear
			less than `min_vocab_times`	will be replaced by `<unk>`. Default: 10.
		max_sen_length (int): All sentences longer than `max_sen_length` will be shortened
			to first `max_sen_length` tokens. Default: 50.
		max_turn_length (int): All sessions longer than `max_turn_length` will be shortened
			to first `max_turn_length` sentences. Default: 20.

	Refer to :class:`.MultiTurnDialog` for attributes and methods.

	Todo:
		* add references
	'''
	def __init__(self, file_path, min_vocab_times=10, max_sen_length=50, max_turn_length=20):
		self._file_path = file_path
		self._min_vocab_times = min_vocab_times
		self._max_sen_length = max_sen_length
		self._max_turn_length = max_turn_length
		super(UbuntuCorpus, self).__init__()

	def _load_data(self):
		r'''Loading dataset, invoked by `MultiTurnDialog.__init__`
		'''
		origin_data = {}
		for key in self.key_name:
			with open('%s/ubuntu_corpus_%s.csv' % (self._file_path, key)) as data_file:
				raw_data = list(csv.reader(data_file))
				head = raw_data[0]
				if head[2] == 'Label':
					raw_data = [d[0] + d[1] for d in raw_data[1:] if d[2] == '1.0']
				else:
					raw_data = [d[0] + d[1] for d in raw_data[1:]]

				raw2line = lambda raw: [sent.strip().split() \
						for sent in raw.strip().replace('__eou__', '<eos>').split('__eot__')]
				origin_data[key] = {'session': list(map(raw2line, raw_data))}

		vocab = list(chain(*chain(*(origin_data['train']['session']))))
		# Important: Sort the words preventing the index changes between different runs
		vocab = sorted(Counter(vocab).most_common(), key=lambda pair: (-pair[1], pair[0]))
		left_vocab = list(filter(lambda x: x[1] >= self._min_vocab_times, vocab))
		left_vocab = list(map(lambda x: x[0], left_vocab))
		left_vocab.remove('<eos>')
		vocab_list = self.ext_vocab + left_vocab
		word2id = {w: i for i, w in enumerate(vocab_list)}
		print("vocab list length = %d" % len(vocab_list))

		line2id = lambda line: ([self.go_id] + list(\
					map(lambda word: word2id.get(word, self.unk_id), line)) + \
					[self.eot_id])[:self._max_sen_length]

		data = {}
		data_size = {}
		for key in self.key_name:
			data[key] = {}
			data[key]['session'] = [list(map(line2id, session[:self._max_turn_length])) \
					for session in origin_data[key]['session']]
			data_size[key] = len(data[key]['session'])
			vocab = list(chain(*chain(*(origin_data[key]['session']))))
			vocab_num = len(vocab)
			oov_num = len(list(filter(lambda word: word not in word2id, vocab)))
			sent_length = list(map(len, chain(*origin_data[key]['session'])))
			cut_word_num = np.sum(np.maximum(np.array(sent_length) - self._max_sen_length + 2, 0))
			turn_length = list(map(len, origin_data[key]['session']))
			sent_num = np.sum(turn_length)
			cut_sent_num = np.sum(np.maximum(np.array(turn_length) - self._max_turn_length, 0))
			print(("%s set. OOV rate: %f, max sentence length before cut: %d, cut word " + \
					"rate: %f\n\tmax turn length before cut: %d, cut sentence rate: %f") % \
					(key, oov_num / vocab_num, max(sent_length), cut_word_num / vocab_num, \
					max(turn_length), cut_sent_num / sent_num))
		return vocab_list, data, data_size
