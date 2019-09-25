'''
A module for dataloader
'''
import random
from collections import Counter

import numpy as np
from nltk.tokenize import WordPunctTokenizer

from .._utils import trim_before_target
from .._utils.metaclass import DocStringInheritor, LoadClassInterface


class Dataloader(LoadClassInterface, metaclass=DocStringInheritor):
	'''Base class of Dataloader.
	'''

	def __init__(self):
		pass


class LanguageProcessingBase(Dataloader):
	r"""Base class for all language processing tasks. This is an abstract class.

	Arguments:{ARGUMENTS}

	Attributes:{ATTRIBUTES}
	"""

	ARGUMENTS = r"""
			ext_vocab (list): special tokens. Default: ``["<pad>", "<unk>", "<go>", "<eos>"]``
			key_name (list): name of subsets of the data. Default: ``["train", "dev", "test"]``"""
	ATTRIBUTES = r"""
			ext_vocab (list): special tokens, placed at beginning of ``vocab_list``.
					For example: ``["<pad>", "<unk>", "<go>", "<eos>"]``.
			pad_id (int): token for padding, always equal to ``0``.
			unk_id (int): token for unknown words, always equal to ``1``.
			go_id (int): token at the beginning of sentences, always equal to ``2``.
			eos_id (int): token at the end of sentences, always equal to ``3``.
			key_name (list): name of subsets of the data. For example: ``["train", "dev", "test"]``.
			all_vocab_list (list): vocabulary list of the datasets,
					including valid vocabs and invalid vocabs.
			word2id (dict): a dict mapping tokens to its id. You don't need to use it 
					at most times, see :meth:`convert_tokens_to_ids` instead."""

	def __init__(self, \
				 ext_vocab=None, \
				 key_name=None):
		super().__init__()

		# initialize by default value. (can be overwritten by subclass)
		self.ext_vocab = ext_vocab or ["<pad>", "<unk>", "<go>", "<eos>"]
		self.pad_id = 0
		self.unk_id = 1
		self.go_id = 2
		self.eos_id = 3
		self.key_name = key_name or ["train", "dev", "test"]

		# initialize by subclass
		self.all_vocab_list, self.valid_vocab_len, self.data, self.data_size = self._load_data()
		self.word2id = {w: i for i, w in enumerate(self.all_vocab_list)}

		# postprocess initialization
		self.index = {}
		self.batch_id = {}
		self.batch_size = {}
		for key in self.key_name:
			self.batch_id[key] = 0
			self.batch_size[key] = None
			self.index[key] = list(range(self.data_size[key]))

	def _valid_word2id(self, word):
		'''This function return the id for a valid word, otherwise return ``unk_id``.

		Arguments:
			word (str): a word.

		Returns:
			int
		'''
		idx = self.word2id.get(word, self.unk_id)
		if idx >= self.vocab_size:
			idx = self.unk_id
		return idx

	def tokenize(self, sentence, remains_capital, tokenizer):
		r'''Convert sentence(str) to a list of tokens(str)

		Arguments:
			sentence (str): a string to be tokenized
			remains_capital(bool): Whether remaining capital letter in data or converting them to lower case.
			tokenizer (str): How to tokenize sentence. ``nltk.tokenize.WordPunctTokenizer`` is used if ``nltk`` is specified,
				python built-in ``str.split`` is used if ``space`` is specified.

		Returns:
			list: a list of tokens(str)
		'''
		if remains_capital:
			sentence = sentence.strip()
		else:
			sentence = sentence.lower().strip()
		if tokenizer == 'nltk':
			return WordPunctTokenizer().tokenize(sentence)
		elif tokenizer == 'space':
			return sentence.split()
		else:
			raise ValueError('tokenizer of dataloader should be either "nltk" or "space"')

	def _general_load_data(self, file_path, data_format, min_vocab_times, max_sent_length, max_turn_length,
						   invalid_vocab_times):
		r'''This function implements a general loading process.

		Arguments:
			file_path (str): A string indicating the path of dataset.
			data_format (list): A list of (key, data_type) pairs, which indicate what the dataset contains. data_type
				must be in ('label', 'sentence', 'session').
				For example, data_format=[['key1', 'sentence'], ['key2', 'label']] means that, in the raw file,
				the first line is a sentence and the second line is a label. They are saved in a dict.
				dataset = {'key1': [line1, line3, line5, ...], 'key2': [line2, line4, line6, ...]}

				data_format=[['key1', 'session], ['key2', 'label']], means that, in the raw file, the first *several lines*
				is a session, *followed by an empty line*, and the next line is a label.
				dataset = {'key1': [session1, session2, ...], 'key2': [label1, label2, ...]}
			min_vocab_times (int): A cut-off threshold of valid tokens. All tokens appear
				not less than `min_vocab_times` in **training set** will be marked as valid words.
			max_sent_length (int): All sentences longer than ``max_sent_length`` will be shortened
				to first ``max_sent_length`` tokens.
			max_turn_length (int): All sessions, whose turn length is longer than ``max_turn_length`` will be shorten to
				first ``max_turn_length`` sentences. If the dataset don't contains sessions, this parameter will be ignored.
			invalid_vocab_times (int):  A cut-off threshold of invalid tokens. All tokens appear
				not less than ``invalid_vocab_times`` in the **whole dataset** (except valid words) will be
				marked as invalid words. Otherwise, they are unknown words, which are ignored both for
				model or metrics.

		Returns:
			(tuple): containing:

			* **all_vocab_list** (list): vocabulary list of the datasets,
			  including valid and invalid vocabs.
			* **valid_vocab_len** (int): the number of valid vocab.
			  ``vocab_list[:valid_vocab_len]`` will be regarded as valid vocabs,
			  while ``vocab_list[valid_vocab_len:]`` regarded as invalid vocabs.
			* **data** (dict): a dict contains data.
			* **data_size** (dict): a dict contains size of each item in data.
		'''
		unknown_type = set(type_ for _, type_ in data_format) - {'label', 'sentence', 'session'}
		if unknown_type:
			raise ValueError('data type must be in ["label", "sentence", "session"]. %s are not allowed' % list(unknown_type))

		def read_sentence(lines):
			return next(lines).rstrip()

		def read_session(lines):
			session = []
			while True:
				try:
					line = next(lines)
					if line == '\n':
						break
					session.append(line.rstrip())
				except StopIteration:
					break
			if not session:
				raise StopIteration
			return session

		def read_label(lines):
			label = int(next(lines).strip())
			return label

		read_functions = {
			'label': read_label,
			'sentence': read_sentence,
			'session': read_session,
		}

		def read(lines, type_):
			return read_functions[type_](lines)

		def to_tokens(element, type_, tokenize):
			if type_ == 'label':
				return element
			if type_ == 'sentence':
				return tokenize(element)
			if type_ == 'session':
				return [tokenize(sentence) for sentence in element]

		def iter_sentence(element, type_):
			if type_ == 'label':
				pass
			elif type_ == 'sentence':
				yield element
			elif type_ == 'session':
				yield from element
			else:
				pass

		def iter_token(element, type_):
			for sentence in iter_sentence(element, type_):
				yield from sentence

		origin_data = {}
		for key in self.key_name:
			origin_data[key] = {data_key: [] for data_key, _ in data_format}
			with open("%s/%s.txt" % (file_path, key), encoding='utf-8') as f_file:
				while True:
					try:
						for data_key, type_ in data_format:
							origin_data[key][data_key].append(to_tokens(read(f_file, type_), type_, self.tokenize))
					except StopIteration:
						break

		def chain_allvocab(dic):
			li = []
			for key, type_ in data_format:
				for element in dic[key]:
					li.extend(iter_token(element, type_))
			return li

		raw_vocab_list = chain_allvocab(origin_data['train'])
		# Important: Sort the words preventing the index changes between
		# different runs
		vocab = sorted(Counter(raw_vocab_list).most_common(), \
					   key=lambda pair: (-pair[1], pair[0]))
		left_vocab = [x[0] for x in vocab if x[1] >= min_vocab_times]
		vocab_list = self.ext_vocab + list(left_vocab)
		valid_vocab_len = len(vocab_list)
		valid_vocab_set = set(vocab_list)

		for key in self.key_name:
			if key == 'train':
				continue
			raw_vocab_list.extend(chain_allvocab(origin_data[key]))

		vocab = sorted(Counter(raw_vocab_list).most_common(), \
					   key=lambda pair: (-pair[1], pair[0]))
		left_vocab = [x[0] for x in vocab if x[1] >= invalid_vocab_times and x[0] not in valid_vocab_set]
		vocab_list.extend(left_vocab)

		print("valid vocab list length = %d" % valid_vocab_len)
		print("vocab list length = %d" % len(vocab_list))

		word2id = {w: i for i, w in enumerate(vocab_list)}

		def line2id(line):
			return [self.go_id] + \
					list(map(lambda word: word2id[word] if word in word2id else self.unk_id, line)) \
					+ [self.eos_id]

		def to_id(element, type_):
			if type_ == 'label':
				return element
			if type_ == 'sentence':
				return line2id(element)
			if type_ == 'session':
				return [line2id(sentence) for sentence in element]

		def cut(element, type_):
			if type_ == 'label':
				return element
			if type_ == 'sentence':
				return element[: max_sent_length]
			if type_ == 'session':
				return [sentence[:max_sent_length] for sentence in element[:max_turn_length]]

		data = {}
		data_size = {}
		for key in self.key_name:
			data[key] = {}
			for data_key, type_ in data_format:
				origin_data[key][data_key] = [to_id(element, type_) for element in origin_data[key][data_key]]
				data[key][data_key] = [cut(element, type_) for element in origin_data[key][data_key]]
				if key not in data_size:
					data_size[key] = len(data[key][data_key])
				elif data_size[key] != len(data[key][data_key]):
					raise RuntimeError(
						"The data of input %s.txt contains different numbers of fields" % key)

			vocab = chain_allvocab(origin_data[key])
			vocab_num = len(vocab)
			oov_num = sum([word not in word2id for word in vocab])
			invalid_num = sum([word not in valid_vocab_set for word in vocab]) - oov_num

			sent_length = []
			for data_key, type_ in data_format:
				if type_ != 'label':
					sent_length.extend([len(sent) for element in origin_data[key][data_key] for sent in iter_sentence(element, type_)])

			cut_word_num = np.sum(np.maximum(np.array(sent_length) - max_sent_length, 0))
			if 'session' in [type_ for _, type_ in data_format]:
				turn_length = list(map(len, origin_data[key]['session']))
				max_turn_length_before_cut = max(turn_length)
				sent_num = sum(turn_length)
				cut_sentence_rate = np.sum(np.maximum(np.array(turn_length) - max_turn_length, 0)) / sent_num
			else:
				max_turn_length_before_cut = 1
				cut_sentence_rate = 0
			print(("%s set. invalid rate: %f, unknown rate: %f, max sentence length before cut: %d, " + \
				   "cut word rate: %f\n\tmax turn length before cut: %d, cut sentence rate: %f") % \
				  (key, invalid_num / vocab_num, oov_num / vocab_num, max(sent_length), \
				   cut_word_num / vocab_num, max_turn_length_before_cut, cut_sentence_rate))
		return vocab_list, valid_vocab_len, data, data_size

	def _load_data(self):
		r'''This function is called during the initialization.

		Returns:
			(tuple): containing:

			* **all_vocab_list** (list): vocabulary list of the datasets,
			  including valid and invalid vocabs.
			* **valid_vocab_len** (int): the number of valid vocab.
			  ``vocab_list[:valid_vocab_len]`` will be regarded as valid vocabs,
			  while ``vocab_list[valid_vocab_len:]`` regarded as invalid vocabs.
			* **data** (dict): a dict contains data.
			* **data_size** (dict): a dict contains size of each item in data.
		'''
		raise NotImplementedError( \
			"This function should be implemented by subclasses.")

	@property
	def vocab_size(self):
		'''int: equals to ``valid_vocab_len``.
		'''
		return self.valid_vocab_len

	@property
	def all_vocab_size(self):
		'''int: equals to ``len(self.all_vocab_list)``.
		'''
		return len(self.all_vocab_list)

	@property
	def vocab_list(self):
		r'''list: valid vocab list, equals to ``all_vocab_list[：valid_vocab_len]``.
		'''
		# using utf-8 ：instead of : for avoiding bug in sphinx
		return self.all_vocab_list[:self.valid_vocab_len]

	def restart(self, key, batch_size=None, shuffle=True):
		r'''Initialize batches. This function be called before :func:`get_next_batch`
		or an epoch is end.

		Arguments:
				key (str): key name of dataset, must be contained in ``self.key_name``.
				batch_size (int): the number of sample in a batch.
					default: if ``None``, last ``batch_size`` is used.
				shuffle (bool): whether to shuffle the data. Default: ``True``.
		'''
		if key not in self.key_name:
			raise ValueError("No set named %s." % key)
		if batch_size is None and self.batch_size[key] is None:
			raise ValueError("You need batch_size to initialize.")
		if shuffle:
			rng_state = random.getstate()
			random.shuffle(self.index[key])
			random.setstate(rng_state)

		self.batch_id[key] = 0
		if batch_size is not None:
			self.batch_size[key] = batch_size
		print("%s set restart, %d batches and %d left" % (key, \
														  len(self.index[key]) // self.batch_size[key], \
														  len(self.index[key]) % self.batch_size[key]))

	GET_BATCH_DOC_WITHOUT_RETURNS = r'''
		Get a batch of specified `indexes`.

		Arguments:
				key (str): key name of dataset, must be contained in ``self.key_name``.
				indexes (list): a list of specified indexes of batched data.
	'''

	def get_batch(self, key, indexes):
		r'''{GET_BATCH_DOC_WITHOUT_RETURNS}

		Returns:
				A dict. See examples in subclasses.
		'''
		raise NotImplementedError( \
			"This function should be implemented by subclasses.")

	def get_next_batch(self, key, ignore_left_samples=False):
		'''Get next batch. It can be called only after Initializing batches (:func:`restart`).

		Arguments:
			key (str): key name of dataset, must be contained in ``self.key_name``.
			ignore_left_samples (bool): If the number of left samples is not equal to
				``batch_size``, ignore them. This make sure all batches have same number of samples.
				Default: ``False``

		Returns:
			A dict like :func:`get_batch`, or None if the epoch is end.
		'''
		if key not in self.key_name:
			raise ValueError("No set named %s." % key)
		if self.batch_size[key] is None:
			raise RuntimeError( \
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

	def get_batches(self, key, batch_size=None, shuffle=True, ignore_left_samples=False):
		'''An iterator over batches. It first call :func:`restart`, and then :func:`get_next_batches`\
			until no more data is available.

		Arguments:
			key (str): key name of dataset, must be contained in ``self.key_name``.
			batch_size (int, optional): default: ``None``.  Use ``batch_size`` by default.
			shuffle (bool): whether to shuffle the data. Default: ``True``.
			ignore_left_samples (bool): If the number of left samples is not equal to
				``batch_size``, ignore them. This make sure all batches have same number of samples.
				Default: ``False``.

		Returns:
			An iterator where each element is like :func:`get_batch`.
		'''
		self.restart(key, batch_size, shuffle)
		while True:
			res = self.get_next_batch(key, ignore_left_samples)
			if res is None:
				break
			yield res

	def get_all_batch(self, key):
		r'''Concatenate all batches to a single dict, where padding will not be applied.
		Exactly, this function called :func:`.get_batch` where ``len(indexes)==1`` multiple times
		and concatenate all the values in the returned dicts.


        Arguments:
            key (str): key name of dataset, must be contained in ``self.key_name``.

        Returns:
            A dict like :func:`get_batch`, but all the values are not padded
            and their type will be converted to list.
        '''
		res = {}
		for idx in self.index[key]:
			batch = self.get_batch(key, [idx])
			for attr, val in batch.items():
				if attr not in res:
					res[attr] = []
				if not isinstance(val, (list, np.ndarray)):
					val = [val]
				res[attr].extend(val)
		return res

	def convert_tokens_to_ids(self, sent, invalid_vocab=False):
		r'''Convert a sentence from string to ids representation.

		Arguments:
			sent (list): a list of string, representing each token of the sentences.
			invalid_vocab (bool): whether to provide invalid vocabs.
					If ``False``, invalid vocabs will be replaced by ``unk_id``.
					If ``True``, invalid vocabs will using their own id.
					Default: ``False``

		Returns:
			(list): a list of ids

		Examples:
			>>> # all_vocab_list = ["<pad>", "<unk>", "<go>", "<eos>", "I", "have",
			>>> #	"been", "to", "China"]
			>>> # vocab_size = 7
			>>> # vocab_list = ["<pad>", "<unk>", "<go>", "<eos>", "I", "have", "been"]
			>>> dataloader.convert_tokens_to_ids(
			...	["<go>", "I", "have", "been", "to", "China", "<eos>"], invalid_vocab=False)
			>>> [2, 4, 5, 6, 1, 1, 3]
			>>> dataloader.convert_tokens_to_ids(
			...	["<go>", "I", "have", "been", "to", "China", "<eos>"], invalid_vocab=True)
			>>> [2, 4, 5, 6, 7, 8, 3]
		'''
		if invalid_vocab:
			return list(map(lambda word: self.word2id.get(word, self.unk_id), sent))
		else:
			return list(map(self._valid_word2id, sent))

	def trim(self, ids):
		r'''Trim a sentence represented by ids. There will be two steps:

			* If there is an end token (``<eos>``) in the sentence,
			  find first end token and abandon words after it (included the end token).
			* ignore ``<pad>`` s at the end of the sentence.

		Arguments:
			ids (list or :class:`numpy.ndarray`): a list of int

		Returns:
			(list): a list of trimmed ids

		Examples:

			>>> # all_vocab_list = ["<pad>", "<unk>", "<go>", "<eos>", "I", "have",
			>>> #	"been", "to", "China"]
			>>> dataloader.trim(
			...	[2, 4, 5, 6, 7, 8, 0, 0, 3, 4, 3, 0])
			... # <go> I have been to China <pad> <pad> <eos> I <eos> <pad>
			>>> [2, 4, 5, 6, 7, 8] # <go> I have been to China
		'''

		ids = trim_before_target(list(ids), self.eos_id)
		idx = len(ids)
		while idx > 0 and ids[idx - 1] == self.pad_id:
			idx -= 1
		ids = ids[:idx]
		return ids

	def convert_ids_to_tokens(self, ids, trim=True):
		'''Convert a sentence from ids to string representation.

		Arguments:
				ids (list): a list of int.
				trim (bool): if True, call :func:`trim` before convertion.

		Returns:
			(list): a list of tokens

		Examples:
			>>> # all_vocab_list = ["<pad>", "<unk>", "<go>", "<eos>", "I", "have",
			>>> #	"been", "to", "China"]
			>>> dataloader.convert_ids_to_tokens(
			...		[2, 4, 5, 6, 7, 8, 3, 0, 0], trim=True)
			>>> ["<go>", "I", "have", "been", "to", "China"]
			>>> dataloader.convert_ids_to_tokens(
			...		[2, 4, 5, 6, 7, 8, 3, 0, 0], trim=False)
			>>> ["<go>", "I", "have", "been", "to", "China", "<eos>", "<pad>", "<pad>"]

		'''
		if trim:
			ids = self.trim(ids)
		return list(map(lambda word: self.all_vocab_list[word], ids))
