'''A module for dataloader'''
import random
from typing import Optional, Any, Union, Sequence, Dict, Tuple, Iterable, List
from collections import Counter
from itertools import chain
import logging
from hashlib import sha256

import numpy as np

from .._utils import trim_before_target
from .._utils.unordered_hash import UnorderedSha256, dumps
from .._utils.file_utils import get_resource_file_path
from .._utils.metaclass import DocStringInheritor, LoadClassInterface
from .tokenizer import BaseTokenizer
from .field import Field, Sentence, _FieldContent, _SentenceContent, _SessionContent, field_from_string
from .vocab import BaseVocab, Vocab
from .context import FieldContext, VocabContext

class Dataloader(LoadClassInterface, metaclass=DocStringInheritor):
	'''Base class of Dataloader.
	'''
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
					at most times, see :meth:`convert_tokens_to_ids` instead.
			tokenizer(Tokenizer): converts a sentence to a list of tokens.
			"""

	@staticmethod
	def simple_create(file_id: str, \
				fields: Union[Sequence[Tuple[str, Union[str, Field]]],\
					 		   Dict[str, Sequence[Tuple[str, Union[str, Field]]]]], \
				*,\
				set_names: Optional[Iterable[str]] = None, \
				tokenizer: Union[BaseTokenizer, str] = None, \
				vocab: Optional[BaseVocab] = None, \
				max_sent_length: Optional[int] = None, \
				max_turn_length: Optional[int] = None, \
				convert_to_lower_letter: bool = False, \
				min_valid_vocab_times: Optional[int] = None, \
				min_invalid_vocab_times: Optional[int] = None, \
				special_appeared_in_data: Optional[bool] = None) -> "LanguageProcessingBase":

		with VocabContext.set_parameters(\
				min_valid_vocab_times=min_valid_vocab_times,\
				min_invalid_vocab_times=min_invalid_vocab_times, \
				special_appeared_in_data=special_appeared_in_data):
			with FieldContext.set_parameters(\
					tokenizer=tokenizer, \
					vocab=vocab, \
					max_sent_length=max_sent_length, \
					max_turn_length=max_turn_length, \
					convert_to_lower_letter=convert_to_lower_letter):
				with FieldContext.set_parameters(tokenizer="space", weak=True):
					return LanguageProcessingBase(file_id, fields, set_names=set_names)

	def __init__(self, file_id: str, \
				 fields: Union[Sequence[Tuple[str, Union[str, Field]]],\
					 		   Dict[str, Sequence[Tuple[str, Union[str, Field]]]]], \
				 *,\
				 set_names: Optional[Iterable[str]] = None, \
				 ):

		# init should initialize the following variables:
		# # self.file_id = file_id
		# # self.file_path = get_resource_file_path(file_id)
		# # self.fields: Dict[str, Sequence[Tuple[str, Field]]] = {}
		# # self.index: Dict[str, List[int]] = {}
		# # self.batch_id: Dict[str, int] = {}
		# # self.batch_size: Dict[str, Optional[int]] = {}
		# # self.vocabs: List[BaseVocab] = []
		# # self.tokenizers: List[BaseVocab] = []
		# # self.data: Dict[str, Dict[str, Any]] = {}
		# # self._raw_data_hash = sha256()
		# # self._data_hash = sha256()
		# # self._vocab_hash = sha256()

		self.file_id = file_id
		self.file_path = get_resource_file_path(file_id)

		field_context = None
		if FieldContext.get("vocab") is None:
			field_context = FieldContext.set_parameters(vocab=Vocab())
		if set_names is None:
			set_names = ["train", "dev", "test"]

		fieldcontents: Dict[str, Sequence[Tuple[str, _FieldContent]]] = {}
		self.fields: Dict[str, Sequence[Tuple[str, Field]]] = {}
		if isinstance(fields, (list, tuple)):
			# for each set_name
			fields = {set_name: fields for set_name in set_names}
		if isinstance(fields, dict):
			# already have set_name
			for set_name, fields_in_one_set in fields.items():
				one_fields, one_fieldcontents = self._fill_field_and_create_content(set_name, fields_in_one_set)
				self.fields[set_name] = one_fields
				fieldcontents[set_name] = one_fieldcontents
		else:
			raise TypeError("Unknown type for fields")

		self._load_data(fieldcontents)

		self.vocabs = self._collect_vocabs_from_fields(self.fields)
		self.default_vocab_id = 0 if len(self.vocabs) == 1 else None
		self.tokenizers = self._collect_tokenizers_from_fields(self.fields)
		self.default_tokenizer_id = 0 if len(self.tokenizers) == 1 else None
		self.default_field_set_name: Optional[str] = None
		self.default_field_id: Optional[int] = None
		self._build_vocabs()

		self._setting_hash = self._create_setting_hash()
		self._vocab_hash = self._create_vocab_hash()
		self.data = self._get_data(fieldcontents)
		self._raw_data_hash, self._data_hash = self._create_data_hash(fieldcontents)
		self.index, self.batch_id, self.batch_size = self._init_batch(fieldcontents)

		if field_context:
			field_context.close()

	def _load_data(self, fieldcontents: Dict[str, Sequence[Tuple[str, _FieldContent]]]):
		for set_name, fieldcontents_in_one_set in fieldcontents.items():
			with open("%s/%s.txt" % (self.file_path, set_name), encoding='utf-8') as f_file:
				line_cnt = 0
				file_iterator = iter(f_file)
				while True:
					try:
						for _, fieldcontent in fieldcontents_in_one_set:
							line_add = fieldcontent.read_next(file_iterator)
							if line_add == 0:
								while True:
									if next(file_iterator):
										raise RuntimeError("the file %s corrupted at line %d" % (set_name, line_cnt))
							line_cnt += line_add
					except StopIteration:
						break

			sample_nums = [fieldcontent.get_data_number() for _, fieldcontent in fieldcontents_in_one_set]
			if not all([sample_num == sample_nums[0] for sample_num in sample_nums]):
				raise RuntimeError("the file %s corrupted at end of the file")

		for _, fieldcontents_in_one_set in fieldcontents.items():
			for _, fieldcontent in fieldcontents_in_one_set:
				fieldcontent.process_before_vocab()

	def _init_batch(self, fieldcontents: Dict[str, Sequence[Tuple[str, _FieldContent]]]) -> \
			Tuple[Dict[str, List[int]], Dict[str, int], Dict[str, Optional[int]]]:
		index: Dict[str, List[int]] = {}
		batch_id: Dict[str, int] = {}
		batch_size: Dict[str, Optional[int]] = {}

		for set_name, fieldcontents_in_one_set in fieldcontents.items():
			sample_nums = [fieldcontent.get_data_number() for _, fieldcontent in fieldcontents_in_one_set]
			batch_id[set_name] = 0
			batch_size[set_name] = None
			index[set_name] = list(range(sample_nums[0]))

		return index, batch_id, batch_size

	def _get_data(self, fieldcontents) -> Dict[str, Dict[str, Any]]:
		data: Dict[str, Dict[str, Any]] = {}
		for set_name, fieldcontents_in_one_set in sorted(fieldcontents.items()):
			data[set_name] = {}
			for field_name, fieldcontent in fieldcontents_in_one_set:
				data[set_name][field_name] = fieldcontent.get_data()
		return data

	def _build_vocabs(self):
		for vocab in self.vocabs:
			vocab.build_vocab()

	def _collect_vocabs_from_fields(self, fields) -> List[BaseVocab]:
		vocabs: List[BaseVocab] = []
		for _, fields_in_one_set in sorted(fields.items()): # sort to keep order
			for _, fields in fields_in_one_set:
				vocab = fields.get_vocab()
				if vocab is not None and vocab not in vocabs:
					vocabs.append(vocab)
		return vocabs

	def _collect_tokenizers_from_fields(self, fields) -> List[BaseTokenizer]:
		tokenizers: List[BaseTokenizer] = []
		tokenizers_setting_hash: List[str] = []
		for _, fields_in_one_set in sorted(fields.items()): # sort to keep order
			for _, field in fields_in_one_set:
				tokenizer = field.get_tokenizer()
				if tokenizer is not None and tokenizer.get_setting_hash() not in tokenizers_setting_hash:
					tokenizers.append(tokenizer)
					tokenizers_setting_hash.append(tokenizer.get_setting_hash())
		return tokenizers

	def _fill_field_and_create_content(self, set_name: str, fields: \
				Sequence[Tuple[str, Union[str, Field]]], \
				) -> \
					Tuple[List[Tuple[str, Field]], List[Tuple[str, _FieldContent]]]:

		fieldcontents: List[Tuple[str, _FieldContent]] = []
		new_fields: List[Tuple[str, Field]] = []

		for name, field_name in fields:
			if isinstance(field_name, str):
				field = field_from_string(field_name)
			elif isinstance(field_name, Field):
				field = field_name
			fieldcontent = field._create(set_name)
			fieldcontents.append((name, fieldcontent))
			new_fields.append((name, field))
		return new_fields, fieldcontents

	def _create_data_hash(self, fieldcontents):
		raw_data_hash = sha256()
		data_hash = sha256()
		for set_name, fieldcontents_in_one_set in sorted(fieldcontents.items()):
			for _, fieldcontent in fieldcontents_in_one_set:
				raw_data_hash.update(dumps(fieldcontent.get_raw_data_hash()))
				data_hash.update(dumps(fieldcontent.get_data_hash()))
		return raw_data_hash.hexdigest(), data_hash.hexdigest()

	def _create_setting_hash(self):
		setting_hash = sha256()
		for _, fields_in_one_set in self.fields.items():
			for _, field in fields_in_one_set:
				setting_hash.update(dumps(field._get_setting_hash(self.tokenizers, self.vocabs)))
		for vocab in self.vocabs:
			setting_hash.update(dumps(vocab.get_setting_hash()))
		for tokenizer in self.tokenizers:
			setting_hash.update(dumps(tokenizer.get_setting_hash()))
		return setting_hash.hexdigest()

	def _create_vocab_hash(self):
		vocab_hash = sha256()
		for vocab in self.vocabs:
			vocab_hash.update(dumps(vocab.get_vocab_hash()))
		return vocab_hash.hexdigest()

	def _get_default_vocab(self) -> BaseVocab:
		if self.default_vocab_id is None:
			raise RuntimeError("The dataloader has multiple vocabs. Use vocab object, \
				or specify the default vocab by set_default_vocab.")
		return self.vocabs[self.default_vocab_id]

	def set_default_vocab(self, obj):
		return self.vocabs.index(obj)

	def _get_default_tokenizer(self):
		if self.default_tokenizer_id is None:
			raise RuntimeError("The dataloader has multiple tokenizers. Use tokenizer object, \
				or specify the default tokenizers by set_default_tokenizer.")
		return self.tokenizers[self.default_tokenizer_id]

	def set_default_tokenizer(self, obj):
		return self.tokenizers.index(obj)

	def _get_default_field(self):
		if self.default_field_id is None or self.default_field_set_name is None:
			raise RuntimeError("The dataloader has multiple fields. Use field object, \
				or specify the default field by set_default_field.")
		return self.fields[self.default_field_set_name][self.default_field_id][1]

	def set_default_field(self, set_name, field_name):
		_ = self.get_field(set_name, field_name)
		self.default_field_set_name = set_name
		for i, (f_name, _) in enumerate(self.fields[set_name]):
			if f_name == field_name:
				self.default_field_id = i
				break

	def get_field(self, set_name, field_name):
		for f_name, f_obj in self.fields[set_name]:
			if f_name == field_name:
				return f_obj
		raise ValueError("No such field named %s" % field_name)

	def get_tokenizer(self, set_name, field_name):
		return self.get_field(set_name, field_name).get_tokenizer()

	def get_vocab(self, set_name, field_name):
		return self.get_vocab(set_name, field_name).get_vocab()

	def get_general_hash(self) -> str:
		general_hash = sha256()
		general_hash.update(dumps(self._raw_data_hash))
		general_hash.update(dumps(self._data_hash))
		general_hash.update(dumps(self._vocab_hash))
		general_hash.update(dumps(self._setting_hash))
		return general_hash.hexdigest()

	def get_raw_data_hash(self) -> str:
		return self._raw_data_hash

	def get_data_hash(self) -> str:
		return self._data_hash

	def get_vocab_hash(self) -> str:
		return self._vocab_hash

	def get_setting_hash(self) -> str:
		return self._setting_hash

	def tokenize(self, sentence):
		r'''Convert sentence(str) to a list of tokens(str)

		Arguments:
			sentence (str): a string to be tokenized
			remains_capital(bool): Whether remaining capital letter in data or converting them to lower case.
			tokenizer (str): How to tokenize sentence. ``nltk.tokenize.WordPunctTokenizer`` is used if ``nltk`` is specified,
				python built-in ``str.split`` is used if ``space`` is specified.

		Returns:
			list: a list of tokens(str)
		'''
		return self._get_default_tokenizer().tokenize(sentence)

	def recover_sentence(self, ids: List[int], remove_special=None, trim=True) -> str:
		return self._get_default_field().recover_sentence(ids, remove_special=remove_special, trim=trim)

	def remove_special_in_ids(self, ids: List[int], remove_special=None, trim=True) -> List[int]:
		return self._get_default_field().remove_special_in_ids(ids, remove_special=remove_special, trim=trim)

	def process_sentences(self, sentences, add_special=True, cut=True, allvocabs=True):
		return self._get_default_field().process_sentences(sentences, \
			add_special=add_special, cut=cut, allvocabs=allvocabs)

	@property
	def vocab_size(self):
		'''int: equals to ``valid_vocab_len``.
		'''
		vocab = self._get_default_vocab()
		try:
			return vocab.valid_vocab_len # type: ignore
		except AttributeError:
			raise AttributeError("default vocab do not have vocab_size") from None

	@property
	def all_vocab_size(self):
		'''int: equals to ``len(self.all_vocab_list)``.
		'''
		vocab = self._get_default_vocab()
		try:
			return vocab.all_vocab_size # type: ignore
		except AttributeError:
			raise AttributeError("default vocab do not have all_vocab_size") from None

	@property
	def vocab_list(self):
		r'''list: valid vocab list, equals to ``all_vocab_list[：valid_vocab_len]``.
		'''
		# using utf-8 ：instead of : for avoiding bug in sphinx
		vocab = self._get_default_vocab()
		try:
			return vocab.vocab_list # type: ignore
		except AttributeError:
			raise AttributeError("default vocab do not have vocab_list") from None

	@property
	def go_id(self):
		vocab = self._get_default_vocab()
		try:
			return vocab.go_id # type: ignore
		except AttributeError:
			raise AttributeError("default vocab do not have go_id") from None

	@property
	def unk_id(self):
		vocab = self._get_default_vocab()
		try:
			return vocab.unk_id # type: ignore
		except AttributeError:
			raise AttributeError("default vocab do not have unk_id") from None

	@property
	def eos_id(self):
		vocab = self._get_default_vocab()
		try:
			return vocab.eos_id # type: ignore
		except AttributeError:
			raise AttributeError("default vocab do not have eos_id") from None

	@property
	def pad_id(self):
		vocab = self._get_default_vocab()
		try:
			return vocab.pad_id # type: ignore
		except AttributeError:
			raise AttributeError("default vocab do not have pad_size") from None

	def get_special_tokens(self):
		return self._get_default_vocab().get_special_tokens()

	def restart(self, key, batch_size=None, shuffle=True):
		r'''Initialize batches. This function be called before :func:`get_next_batch`
		or an epoch is end.

		Arguments:
				key (str): key name of dataset, must be contained in ``self.key_name``.
				batch_size (int): the number of sample in a batch.
					default: if ``None``, last ``batch_size`` is used.
				shuffle (bool): whether to shuffle the data. Default: ``True``.
		'''
		if key not in self.fields:
			raise ValueError("No set named %s." % key)
		if batch_size is None and self.batch_size[key] is None:
			raise ValueError("You need batch_size to initialize.")
		if shuffle:
			# rng_state = random.getstate()
			random.shuffle(self.index[key])
			# random.setstate(rng_state)

		self.batch_id[key] = 0
		if batch_size is not None:
			self.batch_size[key] = batch_size
		batch_size_div = self.batch_size[key]
		assert batch_size_div is not None
		print("%s set restart, %d batches and %d left" % (key, \
						len(self.index[key]) // batch_size_div, \
						len(self.index[key]) % batch_size_div))

	GET_BATCH_DOC_WITHOUT_RETURNS = r'''
		Get a batch of specified `indexes`.

		Arguments:
				key (str): key name of dataset, must be contained in ``self.key_name``.
				indexes (list): a list of specified indexes of batched data.
	'''

	def get_batch(self, set_name: str, indexes: List[int]) -> Dict[str, Any]:
		r'''{GET_BATCH_DOC_WITHOUT_RETURNS}

		Returns:
				A dict. See examples in subclasses.
		'''
		if set_name not in self.fields:
			raise ValueError("No set named %s." % set_name)
		res = {}
		for field_name, field_obj in self.fields[set_name]:
			res.update(field_obj._get_batch(field_name, self.data[set_name][field_name], indexes))
		return res

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
		if key not in self.fields:
			raise ValueError("No set named %s." % key)
		batch_size = self.batch_size[key]
		if batch_size is None:
			raise RuntimeError( \
				"Please run restart before calling this function.")
		batch_id = self.batch_id[key]

		start, end = batch_id * \
					 	batch_size, (batch_id + 1) * batch_size
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
		res: Dict[str, List[Any]] = {}
		for idx in self.index[key]:
			batch = self.get_batch(key, [idx])
			for attr, val in batch.items():
				if attr not in res:
					res[attr] = []
				if not isinstance(val, (list, np.ndarray)):
					val = [val]
				res[attr].extend(val)
		return res

	def convert_tokens_to_ids(self, tokens, add_special=False, allvocabs=True):
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
		return self._get_default_field().convert_tokens_to_ids(tokens, add_special=add_special, allvocabs=allvocabs)

	def trim_in_ids(self, ids):
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

		return self._get_default_field().trim_in_ids(ids)

	def convert_ids_to_tokens(self, ids, remove_special=True, trim=True):
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
		return self._get_default_field().convert_ids_to_tokens(ids, remove_special=remove_special, trim=trim)
