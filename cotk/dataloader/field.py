'''A module for field'''
from typing import Optional, List, Union, Iterator, Tuple, Any, Dict
from itertools import chain
import logging
import hashlib

import numpy as np

from .._utils import trim_before_target, chain_sessions, restore_sessions, is_build_private_docs
from .._utils.metaclass import DocStringInheritor, LoadClassInterface, copy_func, copy_property
from .._utils.unordered_hash import UnorderedSha256, dumps
from .tokenizer import SimpleTokenizer, Tokenizer, PretrainedTokenizer
from .vocab import Vocab, GeneralVocab, PretrainedVocab, SimpleVocab
from .context import FieldContext

RawSentenceType = str
TokenizedSentenceType = List[str]
RawSessionType = List[RawSentenceType]
TokenizedSessionType = List[TokenizedSentenceType]

class Field(LoadClassInterface, metaclass=DocStringInheritor):
	'''A base class of data field, which specify the format of the dataset.
	See :ref:`Field<field_ref>` and :ref:`building a dataloader of customized task<customized_tasks_ref>` for usages.

	Notice :class:`Field` object may be shared between different fields, data sets or dataloader.
	Thus it only defines settings and do NOT stores data.
	'''

	NOT_SPECIFIED_DOCS = r'''
	If any argument is not specified,
	the value will be first retrieved from :class:`FieldContext`. If still ``None``, default
	value will be used.
	'''

	if is_build_private_docs():
		__doc__ += r"""The data is exactly stored in :class:`_FieldContent`."""

	DEFAULT_VOCAB_FROM_MAPPINGS = {
		"train": "train",
		"training": "train",
		"dev": "test",
		"development": "test",
		"valid": "test",
		"validation": "test",
		"test": "test",
		"evaluation": "test"
	}
	'''Dict[str, str]:
			Infer the set type (train, test, or extra)
			from the set name. For example, ``DEFAULT_VOCAB_FROM_MAPPINGS["dev"] == "test"`` means that the words from the "dev" set
			is used for test.
	'''

	def get_vocab(self) -> Optional[Vocab]:
		'''Get :class:`Vocab` object for the field. ``None`` if the field do not have a :class:`Vocab`.
		'''
		return None

	def get_tokenizer(self) -> Optional[Tokenizer]:
		'''Get :class:`Tokenizer` object for the field. ``None`` if the field do not have a :class:`Tokenizer`.
		'''
		return None

	def _create(self, set_name: str) -> "_FieldContent":
		'''Create a :class:`_FieldContent` to store data which have been read.

		Arguments:
			set_name (str): specify the set name for the :class:`_FieldContent`, which may affect the vocab type.

		'''
		raise NotImplementedError

	def _get_setting_hash(self, vocabs) -> str:
		'''Get setting hash for the field. ``vocabs`` are provided by :class:`LanguageProcessing`.
		This function only encode index of vocab, and other settings. It only encode index because
		encode the setting hash of vocabs cannot explain whether a :class:`Vocab` is shared between different vocabs or not.

		Arguments:
			vocabs (list): list of :class:`Vocab`.

		'''
		raise NotImplementedError

	_GET_BATCH_DATA_DOCSTRING = '''data (Any): the data stored in dataloader.'''
	if is_build_private_docs():
		_GET_BATCH_DATA_DOCSTRING = "data (Any): the data returned by :meth:`_FieldContent.get_data`."

	_GET_BATCH_RETURN_VALUE = ''
	_GET_BATCH_EXAMPLE = ''
	def get_batch(self, name: str, data: Dict[str, Any], indexes: List[int]) -> Dict[str, Any]:
		'''Invoked by :meth:`LanguageProcessing.get_batch`, return the batched data specified by this field.
		This function is for INTERNAL USE only, but it shows the data format of the returned batch.

		{_GET_BATCH_RETURN_VALUE}

		Arguments:
			name (str): name of the field.
			{_GET_BATCH_DATA_DOCSTRING}
			indexes (List[int]): the indexes of the data in this batch

		{_GET_BATCH_EXAMPLE}
		'''
		raise NotImplementedError

class _FieldContent(metaclass=DocStringInheritor):
	'''Store the content data of a field.
		Different from :class:`Field`, it won't be shared between fields or dataloader,
		and it can save data.
	'''
	def __init__(self):
		self._original_data: List[Any] = []
		self._raw_data_hash: str
		self._data_hash: str

	_GET_NEXT_ARG = r"""
			dataset (Iterator[str]): An iterator of the data file content.
				Generally, each element is a string, that ends with '\n'.
	"""
	def _get_next(self, dataset: Iterator[str]) -> Tuple[Any, int]:
		'''Read the next data from ``dataset`` and returns a 2-tuple (the data, and the number of elements it read from `dataset`).

		Arguments:{_GET_NEXT_ARG}

		'''
		raise NotImplementedError

	def read_next(self, dataset: Iterator[str]) -> int:
		'''Read the next element from ``dataloader`` and store the elements.
		Returns the number of lines read.

		Arguments:
			dataset (Iterator[str]): An iterator of the data file.
		'''
		if not isinstance(self._original_data, list):
			raise RuntimeError("read_next must be called before get_data")
		sent, lines = self._get_next(dataset)
		if lines != 0:
			self._original_data.append(sent)
		return lines

	def process_before_vocab(self):
		'''This function is called after all elements read, but before building vocabulary.
		'''
		raise NotImplementedError

	def get_data_number(self) -> int:
		'''Get the number of elements in this field.
		'''
		return len(self._original_data)

	def get_data(self) -> Any:
		'''Get the data, which will be stored in the :class:`LanguageProcessing`.
		'''
		raise NotImplementedError

	def get_raw_data_hash(self) -> str:
		'''Return the raw data hash of this field content.
		'''
		return self._raw_data_hash

	def get_data_hash(self) -> str:
		'''Return the data hash of this field content.
		'''
		return self._data_hash

class _SentenceContent(_FieldContent):
	'''Store the content data of :class:`Sentence` field.
		Different from :class:`Field`, it won't be shared between fields or dataloader,
		and it can save data.

	Arguments:
		field (Sentence): The corresponding field of this content.
		vocab_from (str): The type of vocab, must be one of ["train", "test", "extra", "default"]
	'''
	def __init__(self, field: "Sentence", vocab_from: str):
		self.field = field
		self.vocab_from = vocab_from
		self._tmp_tokenized_data: Any = None
		super().__init__()

	def _get_next(self, dataset: Iterator[str]) -> Tuple[str, int]:
		"""read the next sentence and returns a 2-tuple (the sentence and number of elements it reads from `dataset`).
		Note that it may raise StopIteration.

		Arguments:{_FieldContent._GET_NEXT_ARG}

		Examples:
			>>> dataset = iter(["I love NLP.\\n", "Yes I do\\n", "I love deep learning\\n"])
			>>> field_content = _SentenceContent("Sentence", "test")
			>>> field_content._get_next(dataset)
			"I love NLP", 1
			>>> field_content._get_next(dataset)
			"Yes I do", 1
			>>> field_content._get_next(dataset)
			"I love deep learning", 1
		"""
		return next(dataset).rstrip(), 1

	def process_before_vocab(self):
		raw_data_hash = UnorderedSha256()
		for data in self._original_data:
			raw_data_hash.update_data(dumps(data))
		self._raw_data_hash = raw_data_hash.hexdigest()

		self._tmp_tokenized_data = tokenized_sents = self.field.tokenize_sentences(self._original_data)

		data_hash = UnorderedSha256()
		for tokenized_sent in tokenized_sents:
			data_hash.update_data(dumps(tokenized_sent))
		self._data_hash = data_hash.hexdigest()

		self.field.get_vocab().add_tokens(list(chain(*tokenized_sents)), self.vocab_from)

	def get_data(self):
		# allvocabs
		id_data = self.field.process_sentences(self._tmp_tokenized_data)
		return {"id": id_data, "str": self._original_data}

	if is_build_private_docs():
		_GET_BATCH_DATA_DOCSTRING = 'data (Dict[str, Any]): the object returned by :meth:`_SentenceContent.get_data`. '\
			"data['str'] is raw sentences. data['id'] is the ids of tokenized sentences."

class Sentence(Field):
	'''Bases: :class:`.dataloader.Field`

	A field for sentence. This class is a virtual class and the base of
	:class:`Sentence` and :class:`SentenceGPT2`.

	{INIT_DOCSTRING}

	{SENTENCE_INPUT_FORMAT}
	'''

	INIT_DOCSTRING = r'''
	{Field.NOT_SPECIFIED_DOCS}

	Arguments:
			{Sentence.TOKENIZER_DOCS} {Sentence.TOKENIZER_DEFAULT}
			{Sentence.VOCAB_DOCS} {Sentence.VOCAB_DEFAULT}
			{Sentence.VOCAB_FROM_MAPPINGS_DOCS} {Sentence.VOCAB_FROM_MAPPINGS_DEFAULT}
			{Sentence.MAX_SENT_LENGTH_DOCS} {Sentence.MAX_SENT_LENGTH_DEFAULT}
			{Sentence.CONVERT_TO_LOWER_LETTER_DOCS} {Sentence.CONVERT_TO_LOWER_LETTER_DEFAULT}
		'''

	SENTENCE_INPUT_FORMAT = r"""
	Input Formats
		This field read one line of sentence per sample.
	"""

	TOKENIZER_DOCS = r"""
			tokenizer (:class:`Tokenizer`, str, optional): How to tokenize sentence. if ``str``, see :ref:`tokenizer<tokenizer_ref>` for
					possible value."""
	TOKENIZER_DEFAULT = r'''No default value, ``KeyError`` will be raised.'''
	VOCAB_DOCS = r"""
			vocab (:class:`Vocab`, optional):The vocabulary used for this field. Sharing this object between fields can
					build vocabulary together. """
	VOCAB_DEFAULT = r'''No default value, ``KeyError`` will be raised.'''
	VOCAB_FROM_MAPPINGS_DOCS = r"""
			vocab_from_mappings (Dict[str, str], optional): Infer the set type (train, test, or extra) from the set name.
				For example, ``DEFAULT_VOCAB_FROM_MAPPINGS["dev"] == "test"`` means that the words from the "dev" set
				is used for test."""
	VOCAB_FROM_MAPPINGS_DEFAULT = r"""Default: See :ref:`the table<vocab_from_ref>` for default value."""
	MAX_SENT_LENGTH_DOCS = r'''
			max_sent_length (int, optional): All sentences longer than ``max_sent_length`` will be shortened
					to first ``max_sent_length`` tokens.'''
	MAX_SENT_LENGTH_DEFAULT = r'''Default: If ``None``, do not cut the sentence.'''
	CONVERT_TO_LOWER_LETTER_DOCS = r'''
			convert_to_lower_letter (bool, optional): Whether convert all the tokens to lower case after tokenization.'''
	CONVERT_TO_LOWER_LETTER_DEFAULT = r'''Default: ``False``.'''

	def __init__(self, tokenizer: Union[None, Tokenizer, str] = None, \
			vocab: Optional[Vocab] = None, \
			vocab_from_mappings: Optional[Dict[str, str]] = None, \
			max_sent_length: Optional[int] = None, \
			convert_to_lower_letter: Optional[bool] = None):

		if self.__class__.__name__ == "Sentence":
			raise NotImplementedError("Sentence is an abstract class, use SentenceDefault instead.")

		with FieldContext.set_parameters(\
				tokenizer=tokenizer,\
				vocab=vocab,\
				vocab_from_mappings=vocab_from_mappings,\
				max_sent_length=max_sent_length,\
				convert_to_lower_letter=convert_to_lower_letter):
			filled_tokenizer: Union[Tokenizer, str] = FieldContext.get("tokenizer", no_default=True)
			self.vocab: Vocab = FieldContext.get("vocab", no_default=True)
			self.vocab_from_mappings: Dict[str, str] = FieldContext.get("vocab_from_mappings", Field.DEFAULT_VOCAB_FROM_MAPPINGS)
			self.max_sent_length: int = FieldContext.get("max_sent_length", None)
			self.convert_to_lower_letter: bool = FieldContext.get("convert_to_lower_letter", False)

		self.tokenizer: Tokenizer
		if isinstance(filled_tokenizer, str):
			self.tokenizer = SimpleTokenizer(filled_tokenizer)
		elif isinstance(filled_tokenizer, Tokenizer):
			self.tokenizer = filled_tokenizer
		else:
			raise TypeError("Unknown tokenizer type")

	def _create(self, set_name) -> _SentenceContent:
		try:
			return _SentenceContent(self, self.vocab_from_mappings[set_name])
		except KeyError:
			raise KeyError("Unknown set_name %s, do not specify in the vocab_from_mappings" % set_name) from None

	def get_tokenizer(self):
		return self.tokenizer

	def get_vocab(self):
		return self.vocab

	def _get_setting_hash(self, vocabs) -> str:
		return hashlib.sha256(dumps(
			[self.__class__.__name__, \
				#tokenizer_id, \
				self.tokenizer.get_setting_hash(), \
				vocabs.index(self.vocab), \
				#self.vocab.get_setting_hash(), \
				self.vocab_from_mappings, \
				self.max_sent_length, \
				self.convert_to_lower_letter \
			])).hexdigest()

	_SENTENCE_MORE_DOCSTRING = ""
	def tokenize_sentences(self, sentences: List[str]) -> List[List[str]]:
		'''Tokenize sentences and convert them to lower case if ``convert_to_lower_letter`` is ``True``.
		{_SENTENCE_MORE_DOCSTRING}

		Arguments:
			sentences (List[str]): The list of sentence to be tokenized.
		'''
		tokenized_sentences = self.tokenizer.tokenize_sentences(sentences)
		if self.convert_to_lower_letter:
			return [[token.lower() for token in tokens] for tokens in tokenized_sentences]
		else:
			return tokenized_sentences

	def tokenize(self, sentence: str) -> List[str]:
		'''Tokenize sentence and convert it to lower case if ``convert_to_lower_letter`` is ``True``.
		{_SENTENCE_MORE_DOCSTRING}

		Arguments:
			sentence (str): The sentence to be tokenized.
		'''
		tokenized_sentence = self.tokenizer.tokenize(sentence)
		if self.convert_to_lower_letter:
			return [token.lower() for token in tokenized_sentence]
		else:
			return tokenized_sentence

	CONVERT_TO_ID_ARG = r"""
			add_special (bool, optional): If ``True``, special tokens (e.g. ``go``, ``eos``) are added. Default: ``False``.
			only_frequent_word (bool, optional): If ``True``, rare vocabs will be replaced by ``unk_id``. Default: ``False``."""
	def convert_tokens_to_ids(self, tokens: List[str], add_special=False, only_frequent_word=False) -> List[int]:
		'''Convert list of tokens to list of ids. {_SENTENCE_MORE_DOCSTRING}

		Arguments:
			tokens (List[str]): The tokens to be converted.{CONVERT_TO_ID_ARG}
		'''
		ids = self.vocab.convert_tokens_to_ids(tokens, only_frequent_word=only_frequent_word)
		if add_special:
			ids = self.add_special_to_ids(ids)
		return ids

	CONVERT_FROM_ID_ARG = r"""
			remove_special (bool, optional): If ``True``, detect and try to do a reverse operation of ``add_special`` in :meth:`convert_tokens_to_ids`.
					It will not remove ``unk`` or special tokens in the middle of sentences.
					Default: ``True``.
			trim (bool, optional): If ``True``, use :meth:`trim_in_ids` to remove trailing ``pad`` and ``eos``. Default: ``True``."""
	def convert_ids_to_tokens(self, ids: List[int], remove_special=True, trim=True) -> List[str]:
		'''Convert list of ids to list of tokens. {_SENTENCE_MORE_DOCSTRING}

		Arguments:
			ids (List[int]): The ids to be converted.{CONVERT_FROM_ID_ARG}
		'''
		return self.vocab.convert_ids_to_tokens(\
				self.remove_special_in_ids(ids, remove_special=remove_special, trim=trim))

	def convert_ids_to_sentence(self, ids: List[int], remove_special=True, trim=True) -> str:
		'''Convert list of tokens to a sentence. {_SENTENCE_MORE_DOCSTRING}

		Arguments:
			ids (List[int]): The ids to be converted.{CONVERT_FROM_ID_ARG}
		'''
		tokens = self.convert_ids_to_tokens(ids, remove_special=remove_special, trim=trim)
		return self.tokenizer.convert_tokens_to_sentence(tokens)

	def convert_sentence_to_ids(self, sentence: str, add_special=False, only_frequent_word=False) -> List[int]:
		'''Convert a sentence to a list of ids. {_SENTENCE_MORE_DOCSTRING}

		Arguments:
			sentence (str): The sentence to be converted.{CONVERT_TO_ID_ARG}
		'''
		return self.process_sentences([sentence], add_special=add_special, \
				only_frequent_word=only_frequent_word, cut=False)[0]

	def add_special_to_ids(self, ids: List[int]) -> List[int]:
		'''Add special tokens, such as ``go_id`` or ``eos_id`` to the input ``ids``. {_SENTENCE_MORE_DOCSTRING}

		Arguments:
			ids (List[int]): The input ids.
		'''
		raise NotImplementedError

	REMOVE_SPECIAL_ARG = CONVERT_FROM_ID_ARG.replace(":meth:`convert_tokens_to_ids()`", ":meth:`add_special_to_ids`")
	def remove_special_in_ids(self, ids: List[int], remove_special=True, trim=True) -> List[int]:
		'''Remove special ids in input `ids`. {_SENTENCE_MORE_DOCSTRING}

		Arguments:
			ids (List[int]): Input ids.{CONVERT_FROM_ID_ARG}
		'''
		raise NotImplementedError

	PROCESS_ARG = r"""
			add_special (bool, optional): If ``True``, special tokens (e.g. ``go``, ``eos``) are added. Default: ``True``.
			only_frequent_word (bool, optional): If ``True``, rare vocabs will be replaced by ``unk_id``. Default: ``False``."""
	def process_sentences(self, sentences: Union[List[str], List[List[str]]],
						  add_special=True,
						  only_frequent_word=False,
						  cut=True) -> List[List[int]]:
		'''Process input sentences.

		* If sentences haven't been tokenized, tokenize them by invoking :meth:`Sentence.tokenize_sentences`.
		* Then, convert the list of tokens to a list of ids.
		* If ``max_sent_length`` is not ``None`` and ``cut`` is ``True``,
		  sentences, whose length are more than ``self.max_sent_length``, are
		  shorten to first ``max_sent_length`` tokens.

		{_SENTENCE_MORE_DOCSTRING}

		Arguments:
			sentences (List[str], List[List[str]]): `sentences` can be a list of sentences or a list of lists of tokens.
			{PROCESS_ARG}
			cut (bool, optional): Whether to cut sentences with too many tokens. Default: ``True``.
		'''
		# sentences: : Union[List[str], List[List[str]]]
		if not sentences:
			raise ValueError("sentences must not be empty.")
		# list of sentences
		if isinstance(sentences[0], str):
			sentences = self.tokenize_sentences(sentences)
		elif not sentences[0]:
			raise ValueError("sentences[0] must not be an empty string.")

		# list of list of str
		sentences = [self.convert_tokens_to_ids(tokens, add_special=add_special, only_frequent_word=only_frequent_word) for tokens in sentences]
		# list of list of id

		if cut:
			before_lengths = [len(sentence) for sentence in sentences]
			sentences = [sentence[:self.max_sent_length] for sentence in sentences]
			after_lengths = [len(sentence) for sentence in sentences]
			if len(sentences) > 1:
				logging.info("max length before cut: %d, cut percent: %.2f%%" % (
					max(before_lengths),
					(sum(before_lengths) - sum(after_lengths)) / sum(before_lengths) * 100)
							 )
		# sentence cut
		return sentences

	if is_build_private_docs():
		_GET_BATCH_DATA_DOCSTRING = '''data (Any): the object returned by :meth:`_SentenceContent.get_data`'''

	def get_batch(self, name: str, data: Dict[str, Any], indexes: List[int]) -> Dict[str, Any]:
		raise NotImplementedError

	def trim_in_ids(self, ids: List[int]) -> List[int]:
		'''Find the first special token indicating the sentence is over and remove all the tokens after it (included).
		Then remove all trailing ``pad``. {_SENTENCE_MORE_DOCSTRING}

		Arguments:
			ids (List[int]): The input ids.
		'''
		raise NotImplementedError

	def _remove_special_in_ids(self, ids: List[int], go_id: int, eos_id: int) -> List[int]:
		'''Try to remove special token (``go_id`` at the beginning and the ``eos_id`` at the end) in ``ids``.
		{_SENTENCE_MORE_DOCSTRING}

		Arguments:
			ids (List[int]): the original ids
			go_id (int): go token
			eos_id (int): eos token
		'''
		if not ids:
			return ids
		st, ed = 0, None
		if ids[0] == go_id:
			st = 1
		if ids[-1] == eos_id:
			ed = -1
		return ids[st:ed]

	# copy some functions from vocab
	_VOCAB_MORE_DOCSTRING = '''It calls the method with the identical name of the :class:`Vocab` instance, \
		from ``self.get_vocab()``.'''
	frequent_vocab_size = copy_property(get_vocab, Vocab, "frequent_vocab_size")
	all_vocab_size = copy_property(get_vocab, Vocab, "all_vocab_size")
	frequent_vocab_list = copy_property(get_vocab, Vocab, "frequent_vocab_list")
	all_vocab_list = copy_property(get_vocab, Vocab, "all_vocab_list")
	get_special_tokens_mapping = copy_func(get_vocab, Vocab, "get_special_tokens_mapping")
	get_special_tokens_id = copy_func(get_vocab, Vocab, "get_special_tokens_id")
	pad_id = copy_property(get_vocab, Vocab, "pad_id")
	unk_id = copy_property(get_vocab, Vocab, "unk_id")
	go_id = copy_property(get_vocab, Vocab, "go_id")
	eos_id = copy_property(get_vocab, Vocab, "eos_id")

class SentenceDefault(Sentence):
	'''Bases: :class:`.dataloader.Sentence`, :class:`.dataloader.Field`

	A common use field for sentence.

	{INIT_DOCSTRING}

	{SENTENCE_INPUT_FORMAT}
	'''
	INIT_DOCSTRING = Sentence.INIT_DOCSTRING.replace(":class:Vocab", ":class:GeneralVocab")

	def __init__(self, tokenizer: Union[None, Tokenizer, str] = None, \
			vocab: Optional[Vocab] = None, \
			vocab_from_mappings: Optional[Dict[str, str]] = None, \
			max_sent_length: Optional[int] = None, \
			convert_to_lower_letter: Optional[bool] = None):

		super().__init__(tokenizer=tokenizer, \
				vocab=vocab, vocab_from_mappings=vocab_from_mappings, max_sent_length=max_sent_length, \
				convert_to_lower_letter=convert_to_lower_letter)

		self.vocab: Vocab

	def add_special_to_ids(self, ids: List[int]) -> List[int]:
		return [self.vocab.go_id] + ids + [self.vocab.eos_id]

	def remove_special_in_ids(self, ids: List[int], remove_special=True, trim=True) -> List[int]:
		if trim:
			ids = self.trim_in_ids(ids)
		if remove_special:
			ids = self._remove_special_in_ids(ids, self.vocab.go_id, self.vocab.eos_id)
		return ids

	_GET_BATCH_RETURN_VALUE = """
		The function will return a dict, containing:

		* ``FIELDNAME`` (``np.ndarray[batch_size, max_sent_length_in_batch]``):
		  Padded sentences in id formats. It only contains frequent vocabs, and rare words are replaced by ``unk_id``.
		* ``FIELDNAME_allvocabs`` (``np.ndarray[batch_size, max_sent_length_in_batch]``):
		  Padded sentences in id formats. It contains frequent vocabs and rare vocabs.
		* ``FIELDNAME_length`` (``np.ndarray[batch_size]``): The length of sentences.
		* ``FIELDNAME_str`` (``List[str]``): The raw sentences.

		where

		* ``FIELDNAME`` is the name of the field.
		* ``batch_size`` is ``len(indexes)``.
		* ``max_sent_length_in_batch`` is the maximum length of sentences in the batch.
	"""
	_GET_BATCH_EXAMPLE = """
		Examples of the return value:

		.. code-block:: python

			{'sent': array(
				[[ 2, 11,  6, 19,  4,  3,  0,  0,  0,  0,  0],  # <go> Life is short . <eos> <pad> <pad> <pad> <pad> <pad>
				 [ 2, 13,  6, 20, 15, 17, 16, 21,  7,  4,  3]]),# <go> PHP is the best language in the world . <eos>
			'sent_allvocabs': array(
				[[ 2, 11,  6, 19,  4,  3,  0,  0,  0,  0,  0],  # TODO: replace an example have <unk>
				 [ 2, 13,  6, 20, 15, 17, 16, 21,  7,  4,  3]]),
			'sent_length': array([ 6, 11]),
			'sent_str': ['Life is short.', 'PHP is the best language in the world.']}
	"""
	def get_batch(self, name: str, data: Dict[str, Any], indexes: List[int]) -> Dict[str, Any]:
		if not isinstance(self.vocab, GeneralVocab):
			raise RuntimeError("Subclass must override get_batch if self.vocab is not a GeneralVocab.")
		res: Dict[str, Any] = {}
		data_id, data_str = data["id"], data["str"]
		batch_size = len(indexes)
		res[name + "_length"] = np.array([len(data_id[i]) for i in indexes], dtype=int)
		res_sent = res[name] = np.ones((batch_size, np.max(res[name + "_length"])), dtype=int) * self.vocab.pad_id
		for i, j in enumerate(indexes):
			sent = data_id[j]
			res_sent[i, :len(sent)] = sent
		res[name + "_allvocabs"] = res_sent.copy()
		res_sent[res_sent >= self.vocab.frequent_vocab_size] = self.vocab.unk_id
		res[name + "_str"] = [data_str[i] for i in indexes]
		return res

	def trim_in_ids(self, ids: List[int]) -> List[int]:
		ids = trim_before_target(list(ids), self.vocab.eos_id)
		idx = len(ids)
		while idx > 0 and ids[idx - 1] == self.vocab.pad_id:
			idx -= 1
		ids = ids[:idx]
		return ids

class SentenceGPT2(Sentence):
	'''Bases: :class:`.dataloader.Sentence`, :class:`.dataloader.Field`

	A field for sentence in the format of GPT2.

	{INIT_DOCSTRING}

	{SENTENCE_INPUT_FORMAT}
	'''

	INIT_DOCSTRING = Sentence.INIT_DOCSTRING.replace(":class:Vocab", ":class:PretrainedVocab")

	def __init__(self, tokenizer: Union[None, Tokenizer] = None, \
			vocab: Optional[PretrainedVocab] = None, \
			vocab_from_mappings: Optional[Dict[str, str]] = None, \
			max_sent_length: Optional[int] = None, \
			convert_to_lower_letter: Optional[bool] = None):

		super().__init__(tokenizer=tokenizer, \
				vocab=vocab, vocab_from_mappings=vocab_from_mappings,\
				max_sent_length=max_sent_length, \
				convert_to_lower_letter=convert_to_lower_letter)

		if not isinstance(self.tokenizer, PretrainedTokenizer) or self.tokenizer.get_tokenizer_class() != "GPT2Tokenizer":
			raise ValueError("You have to specify a pretrained tokenizer compatible with gpt2")
		self.inner_tokenizer = self.tokenizer.tokenizer

		if not isinstance(self.vocab, PretrainedVocab):
			raise ValueError("You have to specify a PretrainedVocab for SentenceGPT2 field")
		self.vocab: PretrainedVocab

	def add_special_to_ids(self, ids: List[int]) -> List[int]:
		return [self.vocab.eos_id] + ids + [self.vocab.eos_id]

	def remove_special_in_ids(self, ids: List[int], remove_special=True, trim=True) -> List[int]:
		if trim:
			ids = self.trim_in_ids(ids)
		if remove_special:
			ids = self._remove_special_in_ids(ids, self.vocab.eos_id, self.vocab.eos_id)
		return ids

	_GET_BATCH_RETURN_VALUE = SentenceDefault._GET_BATCH_RETURN_VALUE

	_GET_BATCH_EXAMPLE = """
		Examples:
			>>> from transformers.tokenization_gpt2 import GPT2Tokenizer
			>>> field = SentenceDefault('nltk', PretrainedVocab(GPT2Tokenizer.from_pretrained('gpt2')))
			>>> field_content = field._create('train')
			>>> dataset = iter(['I love NLP.', 'Life is short.', 'I use Python.', 'PHP is th best language in the world.', 'Hello, world!'])
			>>> while True:
			... 	try:
			... 		field_content.read_next(dataset)
			... 	except StopIteration:
			... 		break
			>>> field_content.process_before_vocab()
			>>> field.vocab.build_vocab()
			>>> data = field_content.get_data()
			>>> data
			{'id': [[2, 5, 18, 12, 4, 3],
				[2, 11, 6, 19, 4, 3],
				  [2, 5, 22, 14, 4, 3],
				[2, 13, 6, 20, 15, 17, 16, 21, 7, 4, 3],
				[2, 10, 9, 7, 8, 3]],
			 'str': ['I love NLP.',
				  'Life is short.',
				  'I use Python.',
				  'PHP is th best language in the world.',
				  'Hello, world!']}
			>>> # In the above lines, some import variables are defined. It shows how to get `data` and what `data` looks like.
			>>> # When you use :class:`Dataloader`, **you needn't write these codes yourself**. Similar things have been done in :method`LanguageProcessing.__init__`
			>>> field.get_batch('sent', data, [1, 3])
			{'sent_length': array([ 6, 11]),
			  'sent': array([[ 2, 11,  6, 19,  4,  3,  0,  0,  0,  0,  0],
					 [ 2, 13,  6, 20, 15, 17, 16, 21,  7,  4,  3]]),
			  'sent_allvocabs': array([[ 2, 11,  6, 19,  4,  3,  0,  0,  0,  0,  0],
					[ 2, 13,  6, 20, 15, 17, 16, 21,  7,  4,  3]]),
			  'sent_str': ['Life is short.', 'PHP is th best language in the world.']}
			 >>> # 'sent_length' (`name` + '_length') is a :class:`np.ndarray` object with shape == (batch size, ). Each element is the length of corresponding sentence, not including special ids.
			 >>> # 'sent' (`name`) is a :class:`np.ndarray` object with shape == (batch size, max sentence length).
			 >>> # 		Each row is the ids of a sentence. Those sentences, whose length is less than max sentence length, are padded by 0.
			 >>> # 		If an id in the array is rare vocab, it will be replaced be `unk_id`.
			 >>> # 'sent_allvocab' (`name` + '_allvocabs') is the same with 'sent', except that rare id won't be replaced by `unk_id`
			 >>> # 'sent_str' (`name` + '_str') contains the raw sentences.
		"""
	def get_batch(self, name: str, data: Dict[str, Any], indexes: List[int]) -> Dict[str, Any]:
		res: Dict[str, Any] = {}
		data_id, data_str = data["id"], data["str"]
		batch_size = len(indexes)
		res[name + "_length"] = np.array([len(data_id[i]) for i in indexes], dtype=int)
		res_sent = res[name] = np.ones((batch_size, np.max(res[name + "_length"])), dtype=int) * self.vocab.eos_id
		#res_attn = res[name + "_attnmask"] = np.zeros((batch_size, np.max(res[name + "_length"])), dtype=int)
		for i, j in enumerate(indexes):
			sent = data_id[j]
			res_sent[i, :len(sent)] = sent
		#	res_attn[i, :len(sent)] = 1
		res[name + "_allvocabs"] = res_sent.copy()
		res[name + "_str"] = [data_str[i] for i in indexes]
		return res

	def trim_in_ids(self, ids: List[int]) -> List[int]:
		if ids[0] == self.vocab.eos_id:
			ids = [self.vocab.eos_id] + trim_before_target(list(ids[1:]), self.vocab.eos_id)
		else:
			ids = trim_before_target(list(ids), self.vocab.eos_id)
		return ids

class _SessionContent(_FieldContent):
	'''Store the content data of :class:`Session` Field.
		Different from :class:`Field`, it won't be shared between fields or dataloader,
		and it can save data.
	'''
	def __init__(self, field: "Session", vocab_from: str):
		self.field = field
		self.vocab_from = vocab_from
		self._tmp_tokenized_data: Any = None
		super().__init__()

	def _get_next(self, dataset: Iterator[str]) -> Tuple[List[str], int]:
		r"""read **several(one or more)** elements and returns a 2-tuple (the next session, and the number of elements it reads).
		The first several non-space elements in `dataset`, followed by a '\\n', are regarded as a session.
		The first element must not be empty string or '\\n'.
		Note that it may raise StopIteration.

		Arguments:
			{_FieldContent._GET_NEXT_ARG}
		Examples:
			>>> dataset = iter(["a\n", "b\n", "\n", "c\n", "d\e", "e\n", '\n'])
			>>> session_field = "Session"  # For simplicity, `session_field` is a string, rather than a Session object.
			>>> field_content = _SessionContent(session_field, "test")
			>>> field_content._get_next(dataset)
			(['a', 'b'], 2)  # The first session. '\n' separates sessions.
			>>> field_content._get_next(dataset)
			(['c', 'd', 'e'], 3)  # The second(last) session. For the last session, it doesn't matter whether it's followed by '\n'.
		"""
		session: List[str] = []
		lineno = 0
		while True:
			try:
				line = next(dataset)
				lineno += 1
				if line == '\n':
					break
				session.append(line.rstrip())
			except StopIteration:
				break
		if not session:
			raise StopIteration
		return session, lineno

	def process_before_vocab(self):
		raw_data_hash = UnorderedSha256()
		for data in self._original_data:
			raw_data_hash.update_data(dumps(data))
		self._raw_data_hash = raw_data_hash.hexdigest()

		self._tmp_tokenized_data = tokenized_sessions = self.field.tokenize_sessions(self._original_data)

		data_hash = UnorderedSha256()
		for tokenized_data in self._tmp_tokenized_data:
			data_hash.update_data(dumps(tokenized_data))
		self._data_hash = data_hash.hexdigest()

		self.field.get_vocab().add_tokens(list(chain(*chain(*tokenized_sessions))), self.vocab_from)

	def get_data(self) -> Dict[str, list]:
		id_data = self.field.process_sessions(self._tmp_tokenized_data)
		return {"id": id_data, "str": self._original_data}

class Session(Sentence):
	"""Bases: :class:`.dataloader.Field`

	A field for session. Each session is a list of sentences.

	{Sentence.INIT_DOCSTRING}

			max_turn_length (int, optional): Set the maximum turn length of a session. The left sentences are ignored.
				Default: If ``None``, don't cut sessions.

	{SESSION_INPUT_FORMAT}
	"""

	SESSION_INPUT_FORMAT = r"""
	Input Format
		This field read multiple line of sentences per sample, until a blank line.
	"""

	def __init__(self, tokenizer: Union[None, Tokenizer, str] = None,
				 vocab: Optional[Vocab] = None,
				 vocab_from_mappings: Optional[Dict[str, str]] = None,
				 max_sent_length: Optional[int] = None,
				 convert_to_lower_letter: Optional[bool] = None,
				 max_turn_length: Optional[int] = None,):
		if type(self) == Session:
			raise NotImplementedError(
				"%s is an abstract class. Please use %s instead." % (Session.__name__, SessionDefault.__name__))
		super().__init__(tokenizer, vocab, vocab_from_mappings, max_sent_length, convert_to_lower_letter)
		with FieldContext.set_parameters(max_turn_length=max_turn_length):
			max_turn_length = FieldContext.get('max_turn_length')
		if max_turn_length is not None:
			msg = "max_turn_length must be None or a positive integer"
			if not isinstance(max_turn_length, int):
				raise TypeError(msg)
			elif max_turn_length <= 0:
				raise ValueError(msg)
		self.max_turn_length = max_turn_length

	def tokenize_sessions(self, sessions: List[RawSessionType]) -> List[TokenizedSessionType]:
		return [self.tokenize_sentences(session) for session in sessions]

	PROCESS_ARG = Sentence.PROCESS_ARG
	def process_sessions(self, sessions: List[TokenizedSessionType], add_special=True,
						 only_frequent_word=False, cut=True):
		"""Process input sessions.

		* If ``max_turn_length`` is not ``None`` and ``cut`` is ``True``,
		  sessions, whose length are more than ``max_turn_length``, are
		  shorten to first ``max_turn_length`` sentences.
		* If sessions haven’t been tokenized, tokenize them by invoking :meth:`tokenize_sessions`
		* Then, convert the list of tokens to a list of ids.
		* If ``max_sent_length`` is not ``None`` and ``cut`` is ``True``,
		  sentences, whose length are more than ``max_sent_length``, are
		  shorten to first ``max_sent_length`` tokens.

		Arguments:
			sessions (List[List[str], List[List[str]]]):
				sentences in a session can be a str or a list of tokens.
			{PROCESS_ARG}
			cut (bool, optional): Whether to cut sessions/sentences with too many sentences/tokens. Default: ``True``.
		"""
		# Cut sessions.
		# If a session's turn length > `self.max_turn_length`, retain the first `self.max_turn_length` sentences and discard the rest.
		if cut:
			turn_length_before_cut = list(map(len, sessions))
			max_turn_length_before_cut = max(turn_length_before_cut)
			sessions = [session[:self.max_turn_length] for session in sessions]
			turn_length_after_cut = list(map(len, sessions))
			if len(sessions) > 1:
				logging.info("max turn length before cut: %d, cut percent: %.2f%%" % (
					max_turn_length_before_cut,
					100 * (1 - sum(turn_length_after_cut) / sum(turn_length_before_cut)))
							 )

		sentences: List[TokenizedSentenceType]
		session_length: List[int]
		sentences, session_lengths = chain_sessions(sessions)
		processed_sessions = self.process_sentences(sentences, add_special, cut, only_frequent_word)
		processed_sessions = restore_sessions(processed_sessions, session_lengths)
		return processed_sessions

	def _create(self, set_name) -> _SessionContent:
		try:
			return _SessionContent(self, self.vocab_from_mappings[set_name])
		except KeyError:
			raise KeyError("Unknown set_name %s, do not specify in the vocab_from_mappings" % set_name) from None

	def convert_multi_turn_tokens_to_ids(self, session: List[List[str]], add_special=False, only_frequent_word=False) -> \
	List[List[int]]:
		return [self.convert_tokens_to_ids(sent, add_special, only_frequent_word) for sent in session]

	def convert_multi_turn_ids_to_tokens(self, session_ids, remove_special=True, trim=True):
		return [self.convert_ids_to_tokens(sent_ids, remove_special, trim) for sent_ids in session_ids]

	def multi_turn_trim_in_ids(self, session_ids: List[List[int]]) -> List[List[int]]:
		return [self.trim_in_ids(sent_ids) for sent_ids in session_ids]


class SessionDefault(Session):
	'''Bases: :class:`.dataloader.Session`, :class:`.dataloader.Field`

	A common use field for sessions.

	{INIT_DOCSTRING}

	{SESSION_INPUT_FORMAT}
	'''
	INIT_DOCSTRING = Sentence.INIT_DOCSTRING.replace(":class:Vocab", ":class:GeneralVocab")

	add_special_to_ids = SentenceDefault.add_special_to_ids
	remove_special_in_ids = SentenceDefault.remove_special_in_ids
	trim_in_ids = SentenceDefault.trim_in_ids

	_GET_BATCH_DATA_DOCSTRING = SentenceDefault._GET_BATCH_DATA_DOCSTRING.replace(_SentenceContent.__name__, _SessionContent.__name__).replace('sentences', 'sessions')
	_GET_BATCH_EXAMPLE = r"""
	Examples:
		>>> field = SessionDefault('nltk', GeneralVocab())
		>>> field_content = field._create('train')
		>>> dataset = iter(['How are you?\n', "I'm fine. Thank you! And you?\n", "I'm fine, too.\n", "\n", "How to install CoTk?\n", "pip install cotk.\n", "\n"])
		>>> while True:
		... 	try:
		... 		field_content.read_next(dataset)
		... 	except StopIteration:
		... 		break
		>>> field_content.process_before_vocab()
		>>> field.vocab.build_vocab()
		>>> data = field_content.get_data()
		>>> data
		{'id': [[[2, 8, 18, 6, 5, 3],
				[2, 9, 7, 12, 10, 4, 17, 6, 13, 15, 6, 5, 3],
				[2, 9, 7, 12, 10, 14, 22, 4, 3]],
			   [[2, 8, 21, 11, 16, 5, 3], [2, 20, 11, 19, 4, 3]]],
		  'str': [['How are you?', "I'm fine. Thank you! And you?", "I'm fine, too."],
			  ['How to install CoTk?', 'pip install cotk.']]}
		>>> batch_data = field.get_batch('session', data, [1])
		>>> batch_data
		{'session_turn_length': array([2]),
		  'session_sent_length': [[7, 6]],
		  'session': array([[[ 2,  8, 21, 11, 16,  5,  3],
						 [ 2, 20, 11, 19,  4,  3,  0]]]),
		  'session_allvocabs': array([[[ 2,  8, 21, 11, 16,  5,  3],
						 [ 2, 20, 11, 19,  4,  3,  0]]]),
		  'session_str': [['How to install CoTk?', 'pip install cotk.']]}
		>>> # 'session_turn_length' (`name` + '_turn_length') is a :class:`np.ndarray` object with shape == (batch size, ). Each element is the length of corresponding sssion.
		>>> # 'session_sent_length' (`name` + '_sent_length') is List[List[int]]. Each integer is the length of corresponding sentence.
		>>> # 'session' (`name`) is a :class:`np.ndarray` object with shape == (batch size, max turn length, max sentence length).
		>>>				# batch_data['session'][i, j] is a sentence. batch_data['session'][i, j, k] is an id.
		>>>				# If `self.max_turn_length` is not None and j >= `self.max_turn_length` or `self.max_sent_length` is not None and k >= `self.max_sent_length`,
		>>>				# batch_data['session'][i, j, k] is 0.
		>>>				# If an id in the batch_data['session'] is a rare vocab, it will be replaced by `unk_id`.
		>>> # 'session_allvocabs' (`name` + '_allvocabs') is the same with 'session', except that rare vocabs won't be replaced by `unk_id`.
	"""
	def get_batch(self, name: str, data: Dict[str, Any], indexes: List[int]) -> Dict[str, Any]:
		if not isinstance(self.vocab, GeneralVocab):
			raise RuntimeError("Subclass must override get_batch if self.vocab is not a GeneralVocab.")
		res = {}
		data_id, data_str = data['id'], data['str']
		batch_size = len(indexes)
		turn_lengths = res[name + "_turn_length"] = np.array([len(data_id[i]) for i in indexes], dtype=int)
		res[name + "_sent_length"] = [[len(sent) for sent in data_id[i]] for i in indexes]
		max_sent_length = max(map(max, res[name + "_sent_length"]))
		res_session = res[name] = np.zeros((batch_size, turn_lengths.max(), max_sent_length), dtype=int)
		for i, j in enumerate(indexes):
			session = data_id[j]
			session = [list(sent) + [0] * (max_sent_length-len(sent)) for sent in session]
			res_session[i, :len(session)] = np.array(session, dtype=int)
		res[name + "_allvocabs"] = res_session.copy()
		res_session[res_session >= self.vocab.frequent_vocab_size] = self.vocab.unk_id
		res[name + "_str"] = [data_str[i] for i in indexes]
		return res


class DenseLabel(Field):
	"""Bases: :class:`.dataloader.Field`

	A field of categorical labels whose values are integer which
	ranges from ``0`` to ``label_types - 1``.

	See :class:`.dataloader.SparseLabel` for labels in ``str`` or sparse integer.

	Arguments:

		This class do not contains arguments for initialization.

	Input Format
		This field reads one line per sample. The line must be an integer.
	"""
	def _create(self, set_name: str) -> "_FieldContent":
		return _DenseLabelContent(self)

	def _get_setting_hash(self, vocabs) -> str:
		return hashlib.sha256(dumps([self.__class__.__name__])).hexdigest()

	_GET_BATCH_EXAMPLE = r"""
		Examples:
			>>> field = DenseLabel()
			>>> field_content = field._create('train')
			>>> dataset = iter(["1\n", "0\n"])
			>>> while True:
			... 	try:
			... 		field_content.read_next(dataset)
			... 	except StopIteration:
			... 		break
			>>> field_content.process_before_vocab()
			>>> data = field_content.get_data()
			>>> data
			{'label': [1, 0]}
			>>> field.get_batch('label', data, [1])
			{'label': array([0])}  # 'label' (`name`) is a :class:`np.ndarray` object. Each element of it, is a label. 
		"""

	def get_batch(self, name: str, data: Dict[str, Any], indexes: List[int]) -> Dict[str, Any]:
		ids = [data['label'][i] for i in indexes]
		ids = np.array(ids, dtype=int)
		return {name: ids}


class _DenseLabelContent(_FieldContent):
	def __init__(self, field: DenseLabel):
		self.field = field
		super().__init__()

	def _get_next(self, dataset: Iterator[str]) -> Tuple[Any, int]:
		r"""Read the next label and returns a 2-tuple (the next label(integer) and the number of elements it reads).
		Each element in `dataset` represents a label.
		Note that it may raise StopIteration.

		Arguments:{_FieldContent._GET_NEXT_ARG}

		Examples:
			>>> dataset = iter(["1\n", "0\n"])
			>>> field = "DenseLabel"  # For simplicity, field is a string rather than a DenseLabel object.
			>>> field_content = _DenseLabelContent(field)
			>>> field_content.read_next(dataset)
			(1, 1)
			>>> field_content.read_next(dataset)
			(0, 1)

		"""
		label = next(dataset).strip()
		if not label:
			return None, 0
		return int(label), 1

	def get_data(self) -> Any:
		return {"label": self._original_data}

	def process_before_vocab(self):
		raw_data_hash = UnorderedSha256()
		for label in self._original_data:
			raw_data_hash.update_data(dumps(label))
		self._data_hash = self._raw_data_hash = raw_data_hash.hexdigest()

class SparseLabel(Field):
	"""Bases: :class:`.dataloader.Field`

	A field of categorical labels whose values are strings or sparse integer.

	See :class:`.dataloader.DenseLabel` for labels in dense integers.

	{NOT_SPECIFIED_DOCS}

	Arguments:

		vocab (:class:`SimpleVocab`, optional): The vocab to store all the labels.
			If ``None``, a :class:`SimpleVocab` is automatically created.

	Input Format
		This field reads one line per sample. The line can be an arbitary string.
	"""
	def __init__(self, vocab: Optional[SimpleVocab] = None):
		super().__init__()
		self.vocab = vocab if vocab is not None else SimpleVocab()

	def get_vocab(self) -> Optional[Vocab]:
		return self.vocab

	_GET_BATCH_DATA_DOCSTRING = '''data (Dict[str, Any]): the object returned by :meth:`_SparseLabelContent.get_data`.
	 	data['str'] is raw labels.
		data['id'] is ids of labels.
	'''
	_GET_BATCH_EXAMPLE = r"""
		Examples:
			>>> field = SparseLabel()
			>>> field_content = field._create('train')
			>>> dataset = iter(["Java\n", "Python\n", "Cpp\n", "Java\n"])
			>>> while True:
			... 	try:
			... 		field_content.read_next(dataset)
			... 	except StopIteration:
			... 		break
			>>> field_content.process_before_vocab()
			>>> field.get_vocab().build_vocab()
			>>> data = field_content.get_data()
			>>> data
			{'id': [0, 2, 1, 0], 'str': ['Java', 'Python', 'Cpp', 'Java']}
			>>> field.get_batch('label', data, [1])
			{
			 'label_id': array([2]),  # `name` + '_id' is :class`np.ndarray` object with shape == (batch size, ). Each element is the id of corresponding label.
			 'label_str': ['Python']}  # `name` + '_str' is List[str]. Each element is the raw label.

	"""
	def get_batch(self, name: str, data, indexes: List[int]) -> Dict[str, Any]:
		ids = [data['id'][i] for i in indexes]
		ids = np.array(ids, dtype=int)
		batch_size = len(ids)
		return {
			name + "_id": ids,
			name +"_str": [data['str'][i] for i in indexes]
		}

	def _get_setting_hash(self, vocabs) -> str:
		return hashlib.sha256(dumps([self.__class__.__name__])).hexdigest()

	def _create(self, set_name: str) -> "_FieldContent":
		return _SparseLabelContent(self)


class _SparseLabelContent(_FieldContent):
	def __init__(self, field: SparseLabel):
		super().__init__()
		self.field = field

	def _get_next(self, dataset: Iterator[str]) -> Tuple[Union[str, None], int]:
		r"""Read the next label and returns a 2-tuple (the next label(string) and the number of elements it reads).
		Each element in `dataset` represents a label.
		Note that it may raise StopIteration.

		Arguments:{_FieldContent._GET_NEXT_ARG}

		Examples:
			>>> dataset = iter(["Java\n", "Python\n", "Cpp\n", "Java\n"])
			>>> field_content = _SparseLabelContent()
			>>> field_content.read_next(dataset)
			('Java', 1)
			>>> field_content.read_next(dataset)
			('Python', 1)
		"""
		label = next(dataset).rstrip()
		if not label:
			return None, 0
		return label, 1

	def process_before_vocab(self):
		raw_data_hash = UnorderedSha256()
		for label in self._original_data:
			raw_data_hash.update_data(dumps(label))
		self._data_hash = self._raw_data_hash = raw_data_hash.hexdigest()

		self.field.get_vocab().add_tokens(self._original_data, None)

	def get_data(self) -> Any:
		id_data = self.field.get_vocab().convert_tokens_to_ids(self._original_data)
		return {"id": id_data, "str": self._original_data}
