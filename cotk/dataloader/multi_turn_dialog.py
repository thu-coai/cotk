"""
A module for multi turn dialog.
"""
import csv
from collections import Counter
from itertools import chain
import json

import numpy as np

from .._utils.file_utils import get_resource_file_path
from .dataloader import GenerationBase
from ..metric import MetricChain, MultiTurnPerplexityMetric, MultiTurnBleuCorpusMetric, \
	MultiTurnDialogRecorder
from ..metric import BleuPrecisionRecallMetric, EmbSimilarityPrecisionRecallMetric
from ..wordvector import Glove

# pylint: disable=W0223
class MultiTurnDialog(GenerationBase):
	r"""Base class for multi-turn dialog datasets. This is an abstract class.

	Arguments:{ARGUMENTS}

	Attributes:{ATTRIBUTES}
	"""

	ARGUMENTS = GenerationBase.ARGUMENTS
	ATTRIBUTES = GenerationBase.ATTRIBUTES

	def __init__(self, \
				 ext_vocab=None, \
				 key_name=None,	\
		):
		ext_vocab = ext_vocab or ["<pad>", "<unk>", "<go>", "<eos>"]
		super().__init__(ext_vocab, key_name)

	def get_batch(self, key, index):
		'''Get a batch of specified `index`.

		Arguments:
			{ARGUMENTS}

		Returns:
			(dict): A dict at least contains:
			{RETURNSDICT}

			See the example belows.

		Examples:
			{EXAMPLES}

		'''
		if key not in self.key_name:
			raise ValueError("No set named %s." % key)
		res = {}
		res["turn_length"] = [len(self.data[key]['session'][i]) for i in index]
		res["sent_length"] = []
		for i in index:
			sent_length = [len(sent) for sent in self.data[key]['session'][i]]
			res["sent_length"].append(sent_length)
		res_sent = res["sent"] = np.zeros((len(index), np.max(res['turn_length']), \
			np.max(list(chain(*res['sent_length'])))), dtype=int)
		for i, index_i in enumerate(index):
			for j, sent in enumerate(self.data[key]['session'][index_i]):
				res_sent[i, j, :len(sent)] = sent

		res["sent_allvocabs"] = res_sent.copy()
		res_sent[res_sent >= self.valid_vocab_len] = self.unk_id
		return res

	get_batch.ARGUMENTS = r'''
			key (str): must be contained in `key_name`
			index (list): a list of specified index
	'''

	get_batch.RETURNSDICT = r'''
				* turn_length(list): A 1-d list, the number of turns in sessions.
					Size: `[batch_size]`
				* sent_length(list): A 2-d non-padded list, the length of sentence in turns.
					The second dimension is various in different session.
					Length of outer list: `[batch_size]`
				* sent(:class:`numpy.array`): A 3-d padding array containing id of words.
					Only provide valid words. `unk_id` will be used if a word is not valid.
					Size: `[batch_size, max(turn_length[i]), max(sent_length)]`
				* sent_allvocabs(:class:`numpy.array`): A 3-d padding array containing id of words.
					Provide both valid and invalid vocabs.
					Size: `[batch_size, max(turn_length[i]), max(sent_length)]`
	'''

	get_batch.EXAMPLESPART = r'''
			>>> # all_vocab_list = ["<pad>", "<unk>", "<go>", "<eos>", "how", "are", "you",
			>>> #	"hello", "i", "am", "fine"]
			>>> # vocab_size = 9
			>>> # vocab_list = ["<pad>", "<unk>", "<go>", "<eos>", "how", "are", "you", "hello", "i"]
			>>> dataloader.get_batch('train', [0, 1])
			{
				"sent_allvocabs": numpy.array([
					[[2, 7, 3, 0, 0, 0],   # 1st sentence in 1st session: <go> hello <eos> <pad> <pad> <pad>
					[2, 7, 3, 0, 0, 0],    # 2nd sentence in 1st session: <go> hello <eos> <pad> <pad> <pad>
					[2, 4, 5, 6, 3, 0],    # 3rd sentence in 1st session: <go> how are you <eos> <pad>
					[2, 8, 9, 10, 3, 0]],  # 4th sentence in 1st session: <go> i am fine <eos> <pad>
					[[2, 7, 4, 5, 6, 3],   # 1st sentence in 2nd session: <go> hello how are you <eos>
					[2, 8, 9, 10, 3, 0],   # 2nd sentence in 2nd session: <go> i am fine <eos> <pad>
					[0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0]]
				]),
				"sent": numpy.array([
					[[2, 7, 3, 0, 0, 0],  # 1st sentence in 1st session: <go> hello <eos> <pad> <pad> <pad>
					[2, 7, 3, 0, 0, 0],   # 2nd sentence in 1st session: <go> hello <eos> <pad> <pad> <pad>
					[2, 4, 5, 6, 3, 0],   # 3rd sentence in 1st session: <go> how are you <eos> <pad>
					[2, 8, 1, 1, 3, 0]],  # 4th sentence in 1st session: <go> i <unk> <unk> <eos> <pad>
					[[2, 7, 4, 5, 6, 3],  # 1st sentence in 2nd session: <go> hello how are you <eos>
					[2, 8, 1, 1, 3, 0],   # 2nd sentence in 2nd session: <go> i <unk> <unk> <eos> <pad>
					[0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0]]
				]),
				"turn_length": [4, 2], # the number of turns in each session
				"sent_length": [[3, 3, 5, 5], [6, 5]], # length of sentences'''
	get_batch.EXAMPLES = get_batch.EXAMPLESPART + r'''
			}
	'''

	def multi_turn_trim_index(self, index, turn_length=None, ignore_first_token=False):
		'''Trim indexes for multi turn dialog. There will be 3 steps:
			* For every turn, if there is an `<eos>`, \
				find first `<eos>` and abandon words after it (included the `<eos>`).
			* Ignore `<pad>` s at the end of every turn.
			* When `turn_length` is None, discard the first empty turn and the turn after it. \
				Otherwise, discard the turn according to turn_length.

		Arguments:
			index (list or :class:`numpy.array`): a 2-d array of int.
				Size: `[turn_length, max_sent_length]`
			turn_length (int): Default: None
			ignore_first_token (bool): if True, ignore first token of each turn (must be `<go>`).

		Examples:
			>>> # all_vocab_list = ["<pad>", "<unk>", "<go>", "<eos>", "I", "have",
			>>> #	"been", "to", "China", "Japan"]
			>>> dataloader.multi_turn_trim_index(
			...	[[2, 4, 5, 6, 7, 8, 0, 0, 3, 4, 3, 0],
			... 		# <go> I have been to China <pad> <pad> <eos> I <eos> <pad>
			... [2, 4, 5, 6, 7, 9, 3, 0, 3, 4, 3, 0],
			... 		# <go> I have been to Japan <eos> <pad> <eos> I <eos> <pad>
			... [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
			... [2, 4, 5, 6, 7, 8, 0, 0, 3, 4, 3, 0]],
			...			# <go> I have been to China <pad> <pad> <eos> I <eos> <pad>
			... turn_length = None, ignore_first_token = False)
			>>> [[2, 4, 5, 6, 7, 8], [2, 4, 5, 6, 7, 9]]
			>>> dataloader.multi_turn_trim_index(
			...	[[2, 4, 5, 6, 7, 8, 0, 0, 3, 4, 3, 0],
			... [2, 4, 5, 6, 7, 9, 3, 0, 3, 4, 3, 0],
			... [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
			... [2, 4, 5, 6, 7, 8, 0, 0, 3, 4, 3, 0]],
			... turn_length = 1, ignore_first_token = False)
			>>> [[2, 4, 5, 6, 7, 8]]
			>>> dataloader.multi_turn_trim_index(
			...	[[2, 4, 5, 6, 7, 8, 0, 0, 3, 4, 3, 0],
			... [2, 4, 5, 6, 7, 9, 3, 0, 3, 4, 3, 0],
			... [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
			... [2, 4, 5, 6, 7, 8, 0, 0, 3, 4, 3, 0]],
			... turn_length = None, ignore_first_token = True)
			>>> [[4, 5, 6, 7, 8], [4, 5, 6, 7, 9]]
		'''
		res = []
		for i, turn_index in enumerate(index):
			if turn_length and i >= turn_length:
				break
			turn_trim = self.trim_index(turn_index)
			if turn_trim and ignore_first_token:
				turn_trim = turn_trim[1:]
			if turn_length is None and (not turn_trim):
				break
			res.append(turn_trim)
		return res

	def multi_turn_sen_to_index(self, session, invalid_vocab=False):
		'''Convert a session from string to index representation.

		Arguments:
			sen (list): a 2-d list of str, representing each token of the session.
			invalid_vocab (bool): whether to provide invalid vocabs.
					If ``False``, invalid vocabs will be transferred to `unk_id`.
					If ``True``, invalid vocabs will using their own id.
					Default: False

		Examples:
			>>> # all_vocab_list = ["<pad>", "<unk>", "<go>", "<eos>", "I", "have",
			>>> #	"been", "to", "China", "Japan"]
			>>> # vocab_size = 7
			>>> # vocab_list = ["<pad>", "<unk>", "<go>", "<eos>", "I", "have", "been"]
			>>> dataloader.multi_turn_sen_to_index(
			...	[["<go>", "I", "have", "been", "to", "China", "<eos>"],
			... ["<go>", "I", "have", "been", "to", "Japan", "<eos>"]], invalid_vocab = False)
			>>> [[2, 4, 5, 6, 1, 1, 3], [2, 4, 5, 6, 1, 1, 3]]
			>>> dataloader.multi_turn_sen_to_index(
			...	[["<go>", "I", "have", "been", "to", "China", "<eos>"],
			... ["<go>", "I", "have", "been", "to", "Japan", "<eos>"]], invalid_vocab = True)
			>>> [[2, 4, 5, 6, 7 ,8 ,3], [2, 4, 5, 6, 7 ,9 ,3]]
		'''
		if invalid_vocab:
			return list(map(lambda sent: list(map( \
				lambda word: self.word2id.get(word, self.unk_id), sent)), \
			session))
		else:
			return list(map(lambda sent: list(map( \
				self._valid_word2id, sent)), \
			session))

	def multi_turn_index_to_sen(self, index, trim=True, turn_length=None, ignore_first_token=False):
		'''Convert a session from index to string representation

		Arguments:
			index (list or :class:`numpy.array`): a 2-d array of int.
				Size: [turn_length, max_sent_length]
			trim (bool): if True, call :func:`multi_turn_trim_index` before convertion.
			turn_length (int): Only works when trim=True.
				If True, the session is trimmed according the turn_length. Default: None
			ignore_first_token (bool): Only works when trim=True.
				If True, ignore first token of each turn (must be `<go>`).

		Examples:
			>>> # all_vocab_list = ["<pad>", "<unk>", "<go>", "<eos>", "I", "have",
			>>> #	"been", "to", "China", "Japan"]
			>>> dataloader.multi_turn_index_to_sen(
			...	[[2, 4, 5, 6, 7, 8, 3, 0, 0],
			... [2, 4, 5, 6, 7, 9, 3, 0, 0]],
			... trim = True, turn_length = None, ignore_first_token = False)
			>>> [["<go>", "I", "have", "been", "to", "China"],
			... ["<go>", "I", "have", "been", "to", "Japan"]]
			>>> dataloader.multi_turn_index_to_sen(
			...	[[2, 4, 5, 6, 7, 8, 3, 0, 0],
			... [2, 4, 5, 6, 7, 9, 3, 0, 0]],
			... trim = True, turn_length = 1, ignore_first_token = False)
			>>> [["<go>", "I", "have", "been", "to", "China"]]
			>>> dataloader.multi_turn_index_to_sen(
			...	[[2, 4, 5, 6, 7, 8, 3, 0, 0],
			... [2, 4, 5, 6, 7, 9, 3, 0, 0]],
			... trim = True, turn_length = None, ignore_first_token = True)
			>>> [["I", "have", "been", "to", "China"],
			... ["I", "have", "been", "to", "Japan"]]
			>>> dataloader.multi_turn_index_to_sen(
			...	[[2, 4, 5, 6, 7, 8, 3, 0, 0],
			... [2, 4, 5, 6, 7, 9, 3, 0, 0]], trim = False)
			>>> [["<go>", "I", "have", "been", "to", "China", "<eos>", "<pad>", "<pad>"],
			... ["<go>", "I", "have", "been", "to", "Japan", "<eos>", "<pad>", "<pad>"]]
		'''
		if trim:
			index = self.multi_turn_trim_index(index, turn_length=turn_length, \
				ignore_first_token=ignore_first_token)
		return list(map(lambda sent: \
			list(map(lambda word: self.all_vocab_list[word], sent)), \
			index))

	def get_teacher_forcing_metric(self, gen_log_prob_key="gen_log_prob"):
		'''Get metric for teacher-forcing mode.

		It contains:

		* :class:`.metric.MultiTurnPerplexityMetric`

		Arguments:
			gen_prob_key (str): default: `gen_prob`. Refer to :class:`.metric.PerplexityMetric`
		'''
		metric = MetricChain()
		metric.add_metric(MultiTurnPerplexityMetric(self, gen_log_prob_key=gen_log_prob_key))
		return metric

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

	'''A dataloader for Ubuntu dataset.

	Arguments:
		file_id (str): a str indicates the source of UbuntuCorpus dataset.
		file_type (str): a str indicates the type of UbuntuCorpus dataset. Default: "Ubuntu"
		{ARGUMENTS}

	Refer to :class:`.MultiTurnDialog` for attributes and methods.

	References:
		[1] http://dataset.cs.mcgill.ca/ubuntu-corpus-1.0

		[2] https://github.com/rkadlec/ubuntu-ranking-dataset-creator

		[3] Lowe R, Pow N, Serban I, et al. The Ubuntu Dialogue Corpus: A Large Dataset
		for Research in Unstructured Multi-Turn Dialogue Systems. SIGDIAL 2015.
	'''

	ARGUMENTS = r'''
			min_vocab_times (int): A cut-off threshold of `UNK` tokens. All tokens appear
				less than `min_vocab_times`	will be replaced by `<unk>`. Default: 10.
			max_sen_length (int): All sentences longer than `max_sen_length` will be shortened
				to first `max_sen_length` tokens. Default: 50.
			max_turn_length (int): All sessions longer than `max_turn_length` will be shortened
				to first `max_turn_length` sentences. Default: 20.
			invalid_vocab_times (int):  A cut-off threshold of invalid tokens. All tokens appear
				not less than `invalid_vocab_times` in the **whole dataset** (except valid words) will be
				marked as invalid words. Otherwise, they are unknown words, both in training or
				testing stages. Default: 0 (No unknown words).
	'''

	def __init__(self, file_id, min_vocab_times=10, \
			max_sen_length=50, max_turn_length=20, invalid_vocab_times=0):
		self._file_id = file_id
		self._file_path = get_resource_file_path(file_id)
		self._min_vocab_times = min_vocab_times
		self._max_sen_length = max_sen_length
		self._max_turn_length = max_turn_length
		self._invalid_vocab_times = invalid_vocab_times
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
						for sent in raw.strip().replace('__eou__', '').split('__eot__')]
				origin_data[key] = {'session': list(map(raw2line, raw_data))}

		raw_vocab_list = list(chain(*chain(*(origin_data['train']['session']))))
		# Important: Sort the words preventing the index changes between different runs
		vocab = sorted(Counter(raw_vocab_list).most_common(), key=lambda pair: (-pair[1], pair[0]))
		left_vocab = list(filter(lambda x: x[1] >= self._min_vocab_times, vocab))
		left_vocab = list(map(lambda x: x[0], left_vocab))
		vocab_list = self.ext_vocab + left_vocab
		valid_vocab_len = len(vocab_list)
		valid_vocab_set = set(vocab_list)

		for key in self.key_name:
			if key == 'train':
				continue
			raw_vocab_list.extend(list(chain(*chain(*(origin_data[key]['session'])))))
		vocab = sorted(Counter(raw_vocab_list).most_common(), \
					   key=lambda pair: (-pair[1], pair[0]))
		left_vocab = list( \
			filter( \
				lambda x: x[1] >= self._invalid_vocab_times and x[0] not in valid_vocab_set, \
				vocab))
		left_vocab = list(map(lambda x: x[0], left_vocab))
		vocab_list.extend(left_vocab)

		print("valid vocab list length = %d" % valid_vocab_len)
		print("vocab list length = %d" % len(vocab_list))

		word2id = {w: i for i, w in enumerate(vocab_list)}
		line2id = lambda line: ([self.go_id] + list(\
					map(lambda word: word2id.get(word, self.unk_id), line)) + \
					[self.eos_id])[:self._max_sen_length]

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
			invalid_num = len(list(filter(lambda word: word not in valid_vocab_set, vocab))) - oov_num
			sent_length = list(map(len, chain(*origin_data[key]['session'])))
			cut_word_num = np.sum(np.maximum(np.array(sent_length) - self._max_sen_length + 2, 0))
			turn_length = list(map(len, origin_data[key]['session']))
			sent_num = np.sum(turn_length)
			cut_sent_num = np.sum(np.maximum(np.array(turn_length) - self._max_turn_length, 0))
			print(("%s set. invalid rate: %f, unknown rate: %f, max sentence length before cut: %d, " + \
					"cut word rate: %f\n\tmax turn length before cut: %d, cut sentence rate: %f") % \
					(key, invalid_num / vocab_num, oov_num / vocab_num, max(sent_length), \
					cut_word_num / vocab_num, max(turn_length), cut_sent_num / sent_num))
		return vocab_list, valid_vocab_len, data, data_size

class SwitchboardCorpus(MultiTurnDialog):
	'''A dataloader for Switchboard dataset.

	Arguments:
		file_id (str): a str indicates the source of SwitchboardCorpus dataset.
		{ARGUMENTS}

	Refer to :class:`.MultiTurnDialog` for attributes and methods.

	References:
		[1] https://catalog.ldc.upenn.edu/LDC97S62

		[2] John J G and Edward H. Switchboard-1 release 2. Linguistic Data Consortium, Philadelphia 1997.
	'''

	ARGUMENTS = UbuntuCorpus.ARGUMENTS

	def __init__(self, file_id, min_vocab_times=5, max_sen_length=50, max_turn_length=1000, \
				 invalid_vocab_times=0):
		self._file_id = file_id
		self._file_path = get_resource_file_path(file_id)
		self._min_vocab_times = min_vocab_times
		self._max_sen_length = max_sen_length
		self._max_turn_length = max_turn_length
		self._invalid_vocab_times = invalid_vocab_times

		self.word2id = {}
		super().__init__()

	def _convert2ids(self, origin_data):
		'''Convert topic, da, word to ids, invoked by `SwitchboardCorpus._load_data`
		and `SwitchboardCorpus._load_multi_ref_data`

		Arguments:
			origin_data (dict): Contains at least:

				* session (list): A 3-d list, utterances in words.
				  The size of the outermost list is num_data.
				  The size of the second innermost list is the number of utterances in a session.
				  The size of the innermost list is the number of words in a utterance.

		Returns:
			(dict): Contains:

			* session (list): utterances in ids. Size: same as input
		'''
		data = {}
		sess2id = lambda sess: [([self.go_id] + \
								 list(map(lambda word: self.word2id.get(word, self.unk_id), utt)) + \
								 [self.eos_id])[:self._max_sen_length] for utt in sess]
		cand2id = lambda cand: [list(map(lambda word: self.word2id.get(word, self.unk_id), resp)) \
								for resp in cand]
		data['session'] = list(map(sess2id, origin_data['session']))
		if 'candidate_allvocabs' in origin_data:
			data['candidate_allvocabs'] = list(map(cand2id, origin_data['candidate_allvocabs']))
		return data

	def _read_file(self, filepath, read_multi_ref=False):
		'''Reading data from file, invoked by `SwitchboardCorpus._load_data`.

		Arguments:
			filepath (str): Name of the file to read from
			read_multi_ref (bool):
				If False, add turn `<d>` ahead of each session
				If True, add turn `<d>` at the end of each session and read candidate `responses`
		'''
		origin_data = {'session': []}
		if read_multi_ref:
			origin_data['candidate_allvocabs'] = []
		with open(filepath, "r") as data_file:
			for line in data_file:
				line = json.loads(line)
				prefix_utts = [['X', '<d>']] + line['utts']
				# pylint: disable=cell-var-from-loop
				suffix_utts = list(map(lambda utt: utt[1][1].strip() + ' ' \
							if prefix_utts[utt[0]][0] == utt[1][0] \
							else '<eos> ' + utt[1][1].strip() + ' ', enumerate(line['utts'])))
				utts = ('<d> ' + "".join(suffix_utts).strip()).split("<eos>")
				sess = list(map(lambda utt: utt.strip().split(), utts))
				sess = sess[1:] + [['<d>']] if read_multi_ref else sess
				origin_data['session'].append(sess[:self._max_turn_length])

				if read_multi_ref:
					origin_data['candidate_allvocabs'].append(list(map(\
						lambda resp: resp[1].strip().split(), line['responses'])))
		return origin_data

	def _build_vocab(self, origin_data):
		r'''Building vocabulary(words, topics, da), invoked by `SwitchboardCorpus._load_data`
		'''
		raw_vocab = list(chain(*chain(*origin_data['train']['session'])))
		vocab = sorted(Counter(raw_vocab).most_common(), key=lambda pair: (-pair[1], pair[0]))
		left_vocab = list(filter(lambda x: x[1] >= self._min_vocab_times, vocab))
		left_vocab = list(map(lambda x: x[0], left_vocab))
		vocab_list = self.ext_vocab + left_vocab
		self.valid_vocab_len = len(vocab_list)
		valid_vocab_set = set(vocab_list)

		for key in self.key_name:
			if key == 'train':
				continue
			if key == 'multi_ref':
				raw_vocab.extend(list(chain(*chain(*origin_data[key]['candidate_allvocabs']))))
			raw_vocab.extend(list(chain(*chain(*(origin_data[key]['session'])))))
		vocab = sorted(Counter(raw_vocab).most_common(), key=lambda pair: (-pair[1], pair[0]))
		left_vocab = list(filter(lambda x: \
				x[1] >= self._invalid_vocab_times and x[0] not in valid_vocab_set, vocab))
		left_vocab = list(map(lambda x: x[0], left_vocab))
		vocab_list.extend(left_vocab)

		self.word2id = {w: i for i, w in enumerate(vocab_list)}

		print("valid vocab list length = %d" % self.valid_vocab_len)
		print("vocab list length = %d" % len(vocab_list))
		return vocab_list, valid_vocab_set

	def _load_data(self):
		r'''Loading dataset, invoked by `MultiTurnDialog.__init__`
		'''
		origin_data = {}
		self.key_name.append('multi_ref')
		for key in self.key_name:
			origin_data[key] = self._read_file('%s/switchboard_corpus_%s.jsonl' % (self._file_path, key), \
											   read_multi_ref=(key == 'multi_ref'))

		vocab_list, valid_vocab_set = self._build_vocab(origin_data)

		data = {}
		data_size = {s: 0 for s in self.key_name}
		for key in self.key_name:
			data[key] = self._convert2ids(origin_data[key])
			data_size[key] = len(data[key]['session'])

			vocab = list(chain(*chain(*(origin_data[key]['session']))))
			vocab_num = len(vocab)
			oov_num = len(list(filter(lambda word: word not in self.word2id, vocab)))
			invalid_vocab_num = len(list(filter(lambda word: \
											word not in valid_vocab_set, vocab))) - oov_num
			sent_lens = list(map(len, chain(*origin_data[key]['session'])))
			cut_word_num = np.sum(np.maximum(np.array(sent_lens) - self._max_sen_length + 2, 0))
			turn_lens = list(map(len, origin_data[key]['session']))
			cut_sent_num = np.sum(np.maximum(np.array(turn_lens) - self._max_turn_length, 0))
			print(("%s set. invalid rate: %f, unknown rate: %f, max sentence length before cut: %d, " + \
				   "cut word rate: %f\n\tmax turn length before cut: %d, cut sentence rate: %f") % \
				  (key, invalid_vocab_num / vocab_num, oov_num / vocab_num, max(sent_lens), \
				   cut_word_num / vocab_num, max(turn_lens), cut_sent_num / np.sum(turn_lens)))
		return vocab_list, len(valid_vocab_set), data, data_size

	def get_batch(self, key, index):
		'''Get a batch of specified `index`.

		Arguments:
			{ARGUMENTS}

		Returns:
			(dict): A dict contains what is in the return of MultiTurnDialog.get_batch.
				{RETURNSDICT}

				It additionally contains:

				* candidate_allvocabs (list): A 3-d list, multiple responses for reference.
				  The size of outermost list is batch_size, while the size of second innermost list
				  is varying with the number of references and the size of innermost list is varying
				  with the number of words in a reference.

			See the example belows.

		Examples:
			{EXAMPLES}
		'''
		res = super().get_batch(key, index)
		gather = lambda sub_key: [self.data[key][sub_key][i] for i in index]
		for sub_key in self.data[key]:
			if sub_key not in res:
				res[sub_key] = gather(sub_key)
		return res

	get_batch.ARGUMENTS = MultiTurnDialog.get_batch.ARGUMENTS
	get_batch.RETURNSDICT = MultiTurnDialog.get_batch.RETURNSDICT
	get_batch.EXAMPLES = MultiTurnDialog.get_batch.EXAMPLESPART + r'''
				"candidate_allvocabs": [
				[[2, 7, 3],[2, 6, 5, 10, 3]], # two responses to 1st session: <go> hello <eos>
											  # <go> you are fine <eos>
				[[2, 6, 5, 10, 3]]]           # one response to 2nd session: <go> you are fine <eos>
			}
	'''

	def get_precision_recall_metric(self, sent_per_inst=20, embed=None):
		'''Get metrics for precision and recall in terms of BLEU, cosine similarity.

		It contains:

		* :class:`.metric.BleuPrecisionRecallMetric`
		* :class:`.metric.EmbSimilarityPrecisionRecallMetric`

		Arguments:
			sent_per_inst (int): The number of sentences to generate per instance
			embed (:class:`numpy.array`): Word embedding for lookup
				Default: Glove word embedding
		'''
		metric = MetricChain()
		if embed is None:
			glove = Glove("resources://Glove300d")
			embed = glove.load(300, self.vocab_list)
		for ngram in range(1, 5):
			metric.add_metric(BleuPrecisionRecallMetric(self, ngram, sent_per_inst))
		metric.add_metric(EmbSimilarityPrecisionRecallMetric(self, embed, 'avg', sent_per_inst))
		metric.add_metric(EmbSimilarityPrecisionRecallMetric(self, embed, 'extrema', sent_per_inst))
		return metric
