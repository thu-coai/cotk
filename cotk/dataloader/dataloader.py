'''
A module for dataloader
'''
import random
import hashlib
from functools import partial
from collections import Counter
from itertools import chain

import numpy as np
from nltk.tokenize import WordPunctTokenizer

from .._utils import trim_before_target
from .._utils.metaclass import DocStringInheritor, LoadClassInterface
from .._utils.unordered_hash import UnorderedSha256


class DataField(LoadClassInterface, metaclass=DocStringInheritor):
	"""A class that helps process a dataset. It knows the structure of a dataset. Thus, It can get sentences(or sessions,
	or labels, etc) from the raw dataset. It can get all tokens in the dataset and help build a vocabulary list. It can
	convert tokens into ids."""

	@classmethod
	def get_field(cls, field):
		"""If field is an instance of DataField, return it.
		If field is a subclass of DataField, return its instance(assumes that its `__init__` method accepts no arguments).
		If field is a string, we assume it's the name of a subclass of DataField. Search the class and return its instance.

		Args:
			field (str, type, DataField): the data format of dataset.

		Returns　(DataField):

		"""
		if isinstance(field, str):
			field = DataField.load_class(field)
		if isinstance(field, type) and issubclass(field, DataField):
			field = field()
		if not isinstance(field, DataField):
			raise TypeError(
				'argument `field` must be a DataField instance, or a subclass of DataField, or the name(str) of a subclass of DataField.')
		return field

	GET_NEXT_ARG = """
			dataset(Iterator): an Iterator of data.

		Raises:
			StopIteration
	"""

	def get_next(self, dataset):
		"""read **several(one or more)** elements and returns the next data. Note that it may raise StopIteration.

		Args:{GET_NEXT_ARG}
		"""
		return next(dataset)

	def __call__(self, dataset):
		"""Convert an iterator of data to another iterator of data. At each turn, it reads several elements
		from dataset, and transform them to a new format of data, using self.get_next.

		Args:
			dataset (Iterator): an Iterator. Generally speaking, it's an Iterator of strings. But in fact, it can be an
				Iterator of anything.
		"""
		while True:
			try:
				yield self.get_next(dataset)
			except StopIteration:
				break

	CONVERT_TO_TOKENS_ARG = """
			element(str): an element of a raw dataset. It may be a sentence, a session, or something else.
			tokenize(callable): a callable object. If the element is a sentence,
				or a session, it will be used to convert a sentence to a list of tokens."""

	# pylint: disable=W0613
	def convert_to_tokens(self, element, tokenize):
		"""Try to convert an element to tokens, if it consists of several tokens. If the element is a sentence, a list of
		tokens will be returned. If the element is a session, a list of lists of tokens will be returned. By default, it does
		nothing and just return the element.

		Args:{CONVERT_TO_TOKENS_ARG}
		"""
		return element

	# pylint: disable=W0613
	def iter_sentence(self, element):
		"""Returns an generator of the sentences in the element.

		Args:
			element: An element of the dataset. It is returned by the method `convert_to_tokens`.
		"""
		yield from ()

	def iter_tokens(self, element):
		"""Returns an generator of the tokens in the element. This is used for building vocabulary list.

		Args:
			element: An element of the dataset. It is returned by the method `convert_to_tokens`.
		"""
		for sent in self.iter_sentence(element):
			yield from sent

	CONVERT_TO_IDS_ARG = """
			element: An element of the dataset. It is returned by the method `convert_to_tokens`.
			word2id(dict): A dict which maps word(str) to id(int).
			dataloader(Dataloader): A Dataloader, whose attribute `go_id`, `eos_id` are used.
	"""

	def convert_to_ids(self, element, word2id, dataloader):
		"""Convert the element to ids.

		Args:{CONVERT_TO_IDS_ARG}
		"""
		return element

	CUT_ARG = """
			element: An element of the dataset. It is returned by the method `convert_to_ids`."""

	def cut(self, element, **kwargs):
		"""Cut the element if necessary.

		Args:{CUT_ARG}
			**kwargs: Keyword arguments.
		"""
		return element

	@staticmethod
	def convert_obj_to_bytes(obj):
		"""
		Use repr to convert an object to bytes.

		Args:
			obj(object): Any object.

		Returns(bytes):
			Corresponding bytes.
		"""
		return repr(obj).encode()

	def convert_element_to_bytes(self, element, convert_ids_to_tokens):
		"""
		Convert an element to bytes.

		Args:
			element: An element of a dataset.
			convert_ids_to_tokens: A function that converts a list of ids to a list of tokens.

		Returns (bytes):
			Corresponding bytes of element.
		"""
		return self.convert_obj_to_bytes((self.__class__, self._map_fun(element, convert_ids_to_tokens)))

	def _map_fun(self, element, convert_ids_to_tokens):
		"""
		A function that converts an element to another object.
		Generally, if the element is a sentence or a sessions, it will be converted to a list of lists of tokens.

		Args:
			element: An element of a dataset.
			convert_ids_to_tokens: A function that converts a list of ids to a list of tokens.

		Returns:

		"""
		return [convert_ids_to_tokens(sentence) for sentence in self.iter_sentence(element)]


class Sentence(DataField):
	"""Each element in a dataset(Iterator) represent a sentence."""
	def get_next(self, dataset):
		"""read **several(one or more)** elements and returns the next sentence. Note that it may raise StopIteration.

		Args:{DataField.GET_NEXT_ARG}

		Examples:
			>>> dataset = iter(["I\\n", "love\\n", "NLP\\n"])
			>>> field = Sentence()
			>>> field.get_next(dataset)
			"I"
			>>> field.get_next(dataset)
			"love"
			>>> field.get_next(dataset)
			"NLP"
		"""
		return next(dataset).rstrip()

	def convert_to_tokens(self, element, tokenize):
		"""Convert the element(sentence) to a list of tokens.

		Args:{DataField.CONVERT_TO_TOKENS_ARG}
		"""
		return tokenize(element)

	def iter_sentence(self, element):
		yield element

	def convert_to_ids(self, element, word2id, dataloader):
		"""Convert the element(sentence) to ids(a list of integers).

		Args:{DataField.CONVERT_TO_IDS_ARG}"""

		return [dataloader.go_id] + \
					list(map(lambda word: word2id[word] if word in word2id else dataloader.unk_id, element)) \
					+ [dataloader.eos_id]

	# pylint: disable=W0221
	def cut(self, element, max_sent_length=None, **_):
		"""Cut the element(sentence) if it's too long.

		Args:{DataField.CUT_ARG}
			max_sent_length(int): If the length of `element`(sentence) is more than `max_sent_length`,
				it'll be cut off. If `max_sent_length` is None, the sentence won't be cut. Default, None.
		"""
		return element[: max_sent_length] if max_sent_length is not None else element


class Session(DataField):
	"""Several(one or more) elements in a dataset(Iterator), the last one of which is '\\n', represent a session."""
	def get_next(self, dataset):
		r"""read **several(one or more)** elements and returns the next session. The first several non-space elements,
		followed by a '\\n', are regarded as a session. The first element must not be empty string or '\\n'.
		Note that it may raise StopIteration.

		Args:{DataField.GET_NEXT_ARG}

		Examples:
			>>> dataset = iter(["a\n", "b\n", "\n", "c\n", "d\e", "e\n", '\n'])
			>>> field = Session()
			>>> field.get_next(dataset)
			['a', 'b']
			>>> field.get_next(dataset)
			['c', 'd', 'e']
		"""
		session = []
		while True:
			try:
				line = next(dataset)
				if line == '\n':
					break
				session.append(line.rstrip())
			except StopIteration:
				break
		if not session:
			raise StopIteration
		return session

	def convert_to_tokens(self, element, tokenize):
		"""Convert the element(session) to a list of lists of tokens.

		Args:{DataField.CONVERT_TO_TOKENS_ARG}
		"""
		return [tokenize(sentence) for sentence in element]

	def iter_sentence(self, element):
		yield from element

	# pylint: disable=W0221
	def cut(self, element, max_sent_length=None, max_turn_length=None, **_):
		"""Cut the element(element) if it's too long.

		Args:{DataField.CUT_ARG}
			max_sent_length(int): If the length of `element`(sentence) is more than `max_sent_length`,
				it'll be cut off. If `max_sent_length` is None, the sentence won't be cut.
			max_turn_length(int): If the number of sentences in the element(session) is more than `max_turn_length`, the
				session will be cut off. If `max_turn_length` is None, the session won't be cut. Default, None.
		"""
		return [Sentence().cut(sentence, max_sent_length) for sentence in element[: max_turn_length]]

	def convert_to_ids(self, element, word2id, dataloader):
		"""Convert the element(session) to ids(a list of lists of integers).

		Args:{DataField.CONVERT_TO_IDS_ARG}"""
		return [Sentence().convert_to_ids(sentence, word2id, dataloader) for sentence in element]


class Label(DataField):
	"""Each element in a dataset(Iterator) represents a label."""
	def get_next(self, dataset):
		r"""read text and returns the next label(integer). Note that it may raise StopIteration.

		Args:{DataField.GET_NEXT_ARG}

		Examples:
			>>> dataset = iter(["1\n", "0\n"])
			>>> field = Label()
			>>> field.get_next(dataset)
			1
			>>> field.get_next(dataset)
			0
		"""
		label = next(dataset)
		return int(label.strip())

	def _map_fun(self, element, convert_ids_to_tokens=None):
		"""
		Returns the element itself.

		Args:
			element: An element of a dataset.
			convert_ids_to_tokens: It's useless. This argument exists, just to keep the signature the same as that of super class.
		"""
		return element


class Dataloader(LoadClassInterface, metaclass=DocStringInheritor):
	'''Base class of Dataloader.
	'''

	def __init__(self):
		pass


class DataloaderHash(metaclass=DocStringInheritor):
	"""
	A class that can calculate hash value for a dataloader.
	"""
	def __init__(self, ignore_tokens, unk_id=None):
		"""
		Initialize.

		Args:
			ignore_tokens (Iterable): Iterable of integers. Each of them represent an id of a token.
				All these tokens are ignored, when calculating hash value.
			unk_id (int): Id of unknown token(`unk`). If it's None, we assume that there's no `unk` in dataset.
		"""
		self.ignore_tokens = set(ignore_tokens)
		self.unk_id = unk_id
		if unk_id is not None and not isinstance(unk_id, int):
			raise TypeError('`unk_id` must be None, or an integer.')
		for i in self.ignore_tokens:
			if not isinstance(i, int):
				raise ValueError(
					'`ignore_tokens`must be an Iterable of integers, but contains an {}.'.format(i.__class__))

	def convert_ids_to_tokens(self, sentence, id_to_word):
		"""
		Convert a sentence to a list of tokens.

		Args:
			sentence (Iterable): An Iterable object, that contains ids(integers).
			id_to_word (dict, list): An object that has method `__getitem__` and can map integers to strings.

		Returns (list):
			A list of tokens.
		"""
		unknown_token = None
		invalid_token = (None, None)
		tokens = []
		for id_ in sentence:
			if id_ == self.unk_id:
				tokens.append(unknown_token)
			elif id_ in self.ignore_tokens:
				continue
			else:
				try:
					token = id_to_word[id_]
					tokens.append(token)
				except (KeyError, IndexError):
					tokens.append(invalid_token)
		return tokens

	__HASH_DATASET_DOC = \
		"""
		Calculate the hash value of a dataset.

		Args:
			dataset (dict): A dataset that contains several fields.
			fields (dict, list, tuple): If it's a dict, it maps a data_key to a field. If it's a list, it must be a list of lists.
			id_to_word (dict, list): An object that has method `__getitem__` and can map integers to strings.

		"""

	def _hash_dataset(self, dataset, fields, id_to_word):
		if isinstance(fields, list) or isinstance(fields, tuple):
			for item in fields:
				if not ((isinstance(item, list) or isinstance(item, tuple)) and len(item) == 2):
					raise ValueError(
						"If `fields` is a list(tuple), each element of it must be a list(tuple) with length==2.")
			fields = dict(fields)
		elif isinstance(fields, dict):
			pass
		else:
			raise TypeError("`fields` must be a dict, or a lit(tuple) of lists(tuples).")

		convert_ids_to_tokens = partial(self.convert_ids_to_tokens, id_to_word=id_to_word)

		ordered_hash_obj = hashlib.sha256()
		for data_key in sorted(dataset.keys()):
			if data_key not in fields:
				raise ValueError('There isn\'t corresponding field for %s.' % data_key)
			field = fields[data_key]
			field = DataField.get_field(field)
			unordered_hash_obj = UnorderedSha256()
			for element in dataset[data_key]:
				unordered_hash_obj.update_data(field.convert_element_to_bytes(element, convert_ids_to_tokens))
			ordered_hash_obj.update(unordered_hash_obj.result.tobytes())
		return ordered_hash_obj.digest()

	_hash_dataset.__doc__ = __HASH_DATASET_DOC + \
		"""
		Returns (bytes):
			hash value(length==32)
		"""

	def hash_dataset(self, dataset, fields, id_to_word):
		return self._hash_dataset(dataset, fields, id_to_word).hex()

	hash_dataset.__doc__ = __HASH_DATASET_DOC + \
		"""
		Returns (str):
			hex hash value(length==64) of the dataset
		"""
	del __HASH_DATASET_DOC

	def hash_datasets(self, datasets, field_dict, id_to_word):
		"""
		Calculate the hash value of several datasets.

		Args:
			datasets (dict): Several datasets.
			field_dict (dict): A dict that maps a key of a dataset to its fields.
			id_to_word (dict, list): An object that has method `__getitem__` and can map integers to strings.

		Returns (str):
			hex hash value(length==64) of datasets
		"""
		hash_obj = hashlib.sha256()
		for key in sorted(datasets.keys()):
			hash_obj.update(self._hash_dataset(datasets[key], field_dict[key], id_to_word))
		return hash_obj.hexdigest()


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
		self.__hash_value = None  # it's assigned in `_general_load`

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

	@property
	def hash_value(self):
		"""return hash value of dataset"""
		return self.__hash_value

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

	def _general_load_data(self, file_path, data_fields, min_vocab_times, max_sent_length, max_turn_length,
						   invalid_vocab_times):
		r'''This function implements a general loading process.

		Arguments:
			file_path (str): A string indicating the path of dataset.
			data_fields (dict, list, tuple): If it's a list(tuple), it must be a list of (key, field) pairs.
				Field must be a DataField instance,
				or a subclass of DataField(in this case, its instance will be used, assuming its constructor accepts no arguments),
				or a string(in this case, the instance of the class, whose __name__ is field, will be used).

				For example, data_fields=[['post', 'Sentence'], ['label', Label]] means that,
				in the raw file, the first line is a sentence and the second line is a label. They are saved in a dict.
				dataset = {'post': [line1, line3, line5, ...], 'label': [line2, line4, line6, ...]}

				data_fields=[['key1', 'Session'], ['key2', Label()]], means that, in the raw file, the first *several lines*
				is a session, *followed by an empty line*, and the next line is a label.
				dataset = {'key1': [session1, session2, ...], 'key2': [label1, label2, ...]}

				If it's a dict, different datasets may have different formats.(If `data_fields` is a list or a tuple, different datasets have the same format).
				Its keys are the same as `self.key_name` that indicate the datasets, and the values are lists as mentioned above.
				For example, data_fields = {'train': [['sess', 'Session'], ['label', 'Label']], 'test': [['sess', 'session']]},
				means that the train set contains sessions and labels, but the test set only contains sessions.

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
		def get_fields(fields):
			assert isinstance(fields, list) or isinstance(fields, tuple)
			return [(data_key, DataField.get_field(field)) for data_key, field in fields]

		if isinstance(data_fields, dict):
			no_field_keys = [key for key in self.key_name if key not in data_fields]
			if no_field_keys:
				raise ValueError('There is no data fields for dataset(%s) ' % ', '.join(no_field_keys))
			try:
				data_fields = {key: get_fields(data_fields[key]) for key in self.key_name}
			except AssertionError:
				raise TypeError('If `data_field` is a dict, its value must be a list(or tuple) of lists(or tuples).')
		elif isinstance(data_fields, list) or isinstance(data_fields, tuple):
			data_fields = get_fields(data_fields)
			data_fields = {key: data_fields for key in self.key_name}
		else:
			raise TypeError('`data_fields` must be a dict, or a list, or a tuple.')

		# now data_fields is a dict. Keys are the same as self.key_name('train', 'test', 'dev', etc.). Each value is
		# a list(tuple) of lists(tuples), which means (data_key(str), data_field(DataField)) pairs.
		# For example,
		# data_fields == {'train': [['sent', Sentence()], ['label', Label()]],
		# 'test': [['sent', Sentence()], ['label', Label()]]}.
		# Note, different dataset may have different fields.

		special_tokens = set(self.ext_vocab)
		origin_data = {}
		for key in self.key_name:
			origin_data[key] = {data_key: [] for data_key, _ in data_fields[key]}
			with open("%s/%s.txt" % (file_path, key), encoding='utf-8') as f_file:
				while True:
					try:
						for data_key, field in data_fields[key]:
							element = field.convert_to_tokens(field.get_next(f_file), self.tokenize)
							for token in field.iter_tokens(element):
								if token in special_tokens:
									raise RuntimeError('The dataset contains special token "%s". This is not allowed.' % token)
							origin_data[key][data_key].append(element)
					except StopIteration:
						break

		def chain_allvocab(dic, fields):
			vocabs = []
			for data_key, field in fields:
				for element in dic[data_key]:
					vocabs.extend(field.iter_tokens(element))
			return vocabs

		raw_vocab_list = chain_allvocab(origin_data['train'], data_fields['train'])
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
			raw_vocab_list.extend(chain_allvocab(origin_data[key], data_fields[key]))

		vocab = sorted(Counter(raw_vocab_list).most_common(), \
					   key=lambda pair: (-pair[1], pair[0]))
		left_vocab = [x[0] for x in vocab if x[1] >= invalid_vocab_times and x[0] not in valid_vocab_set]
		vocab_list.extend(left_vocab)

		print("valid vocab list length = %d" % valid_vocab_len)
		print("vocab list length = %d" % len(vocab_list))

		word2id = {w: i for i, w in enumerate(vocab_list)}

		data = {}
		data_size = {}
		for key in self.key_name:
			data[key] = {}
			for data_key, field in data_fields[key]:
				origin_data[key][data_key] = [field.convert_to_ids(element, word2id, self) for element in origin_data[key][data_key]]
				data[key][data_key] = [
					field.cut(element, max_sent_length=max_sent_length, max_turn_length=max_turn_length) for element in
					origin_data[key][data_key]]
				if key not in data_size:
					data_size[key] = len(data[key][data_key])
				elif data_size[key] != len(data[key][data_key]):
					raise RuntimeError(
						"The data of input %s.txt contains different numbers of fields" % key)

			vocab = chain_allvocab(origin_data[key], data_fields[key])
			vocab_num = len(vocab)
			oov_num = sum([word not in word2id for word in vocab])
			invalid_num = sum([word not in valid_vocab_set for word in vocab]) - oov_num

			sent_length = []
			for data_key, field in data_fields[key]:
				sent_length.extend([len(sent) for element in origin_data[key][data_key] for sent in field.iter_sentence(element)])

			cut_word_num = np.sum(np.maximum(np.array(sent_length) - max_sent_length, 0))

			session_keys = [data_key for data_key, field in data_fields[key] if field.__class__ == Session]
			if session_keys:
				turn_length = list(map(len, chain.from_iterable((origin_data[key][sess_key] for sess_key in session_keys))))
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

		# calculate hash value
		hash_value = DataloaderHash(ignore_tokens=(self.go_id, self.eos_id, self.pad_id),
									unk_id=self.unk_id).hash_datasets(data, data_fields, vocab_list[len(
			self.ext_vocab):valid_vocab_len])
		self.__hash_value = hash_value

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
