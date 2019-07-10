'''
A module for dataloader
'''
import random
from .._utils import trim_before_target
from .._utils.metaclass import DocStringInheritor, LoadClassInterface
from ..metric import MetricChain, PerplexityMetric, BleuCorpusMetric, SingleTurnDialogRecorder

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
			key_name (list): name of subsets of the data. Default: ``["train", "dev", "test"]``
	"""
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
			word2id (dict): a dict mapping all vocab to index. You don't need to use it 
					at most times, see :meth:`convert_tokens_to_ids` instead.
	"""

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
			random.shuffle(self.index[key])

		self.batch_id[key] = 0
		if batch_size is not None:
			self.batch_size[key] = batch_size
		print("%s set restart, %d batches and %d left" % (key, \
				len(self.index[key]) // self.batch_size[key], \
				len(self.index[key]) % self.batch_size[key]))

	def get_batch(self, key, index):
		'''Get a batch of specified `index`.

		Arguments:
				key (str): key name of dataset, must be contained in ``self.key_name``.
				index (list): a list of specified index.

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

	def convert_tokens_to_ids(self, sent, invalid_vocab=False):
		r'''Convert a sentence from string to index representation.

		Arguments:
			sent (list): a list of string, representing each token of the sentences.
			invalid_vocab (bool): whether to provide invalid vocabs.
					If ``False``, invalid vocabs will be replaced by ``unk_id``.
					If ``True``, invalid vocabs will using their own id.
					Default: ``False``

		Returns:
			(list): a list of indexes

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

	def trim_index(self, index):
		r'''Trim a sentence represented by index. There will be two steps:

			* If there is an end token (``<eos>``) in the sentence,
			  find first end token and abandon words after it (included the end token).
			* ignore ``<pad>`` s at the end of the sentence.

		Arguments:
			index (list or :class:`numpy.ndarray`): a list of int

		Returns:
			(list): a list of trimmed index

		Examples:

			>>> # all_vocab_list = ["<pad>", "<unk>", "<go>", "<eos>", "I", "have",
			>>> #	"been", "to", "China"]
			>>> dataloader.trim_index(
			...	[2, 4, 5, 6, 7, 8, 0, 0, 3, 4, 3, 0])
			... # <go> I have been to China <pad> <pad> <eos> I <eos> <pad>
			>>> [2, 4, 5, 6, 7, 8] # <go> I have been to China
		'''

		index = trim_before_target(list(index), self.eos_id)
		idx = len(index)
		while idx > 0 and index[idx-1] == self.pad_id:
			idx -= 1
		index = index[:idx]
		return index

	def convert_ids_to_tokens(self, index, trim=True):
		'''Convert a sentence from index to string representation.

		Arguments:
				index (list): a list of int.
				trim (bool): if True, call :func:`trim_index` before convertion.

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
			index = self.trim_index(index)
		return list(map(lambda word: self.all_vocab_list[word], index))


try:
	from pytorch_pretrained_bert import BertTokenizer
	import numpy as np
except ImportError as err:
	from .._utils.imports import DummyObject
	BertTokenizer = DummyObject(err)

class BERTLanguageProcessingBase(LanguageProcessingBase):
	r"""Base class for all BERT-based language processing with BERT tokenizer.
	This is an abstract class.

	Arguments:{ARGUMENTS}

	Attributes:{ATTRIBUTES}
	"""

	ARGUMENTS = LanguageProcessingBase.ARGUMENTS
	ATTRIBUTES = LanguageProcessingBase.ATTRIBUTES

	def __init__(self, \
					key_name=None, \
					bert_vocab='bert-base-uncased'):

		# initialize by default value. (can be overwritten by subclass)
		self.ext_vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]"]

		self.tokenizer = BertTokenizer.from_pretrained(bert_vocab)
		self._build_bert_vocab()

		self.bert_data = {}

		super().__init__(self.ext_vocab, key_name)

		self.pad_id = self.ext_vocab.index("[PAD]")
		self.unk_id = self.ext_vocab.index("[UNK]")
		self.go_id = self.ext_vocab.index("[CLS]")
		self.eos_id = self.ext_vocab.index("[SEP]")

		self._build_ids_to_bert_ids()

	def _build_bert_vocab(self):
		self.bert_word2id = dict(self.tokenizer.vocab)
		self.bert_id2word = [None] * len(self.bert_word2id)
		for key, value in self.bert_word2id.items():
			self.bert_id2word[value] = key

		self.bert_pad_id = self.bert_word2id["[PAD]"]
		self.bert_unk_id = self.bert_word2id["[UNK]"]
		self.bert_go_id = self.bert_word2id["[CLS]"]
		self.bert_eos_id = self.bert_word2id["[SEP]"]

	def _build_ids_to_bert_ids(self):
		self.id2bertid = {}
		self.bertid2id = {}
		for key, value in self.bert_word2id.items():
			self.id2bertid[self.word2id[key]] = value
			self.bertid2id[value] = self.word2id[key]

	def tokenize(self, sentence):
		'''Convert sentence(str) to list of token(str)

		Arguments:
				sentence (str)

		Returns:
				sent (list): list of token(str)
		'''
		return self.tokenizer.tokenize(sentence)

	def convert_tokens_to_bert_ids(self, sent):
		'''Convert list of token(str) to list of bert id(int)

		Arguments:
				sent (list): list of token(str)

		Returns:
				bert_ids (list): list of bert id(int)
		'''
		return self.tokenizer.convert_tokens_to_ids(sent)

	def convert_bert_ids_to_ids(self, sent, invalid_vocab=False):
		'''Convert list of bert id(int) to list of id(int)

		Arguments:
				sent (list): list of bert id(int)

		Returns:
				ids (list): list of id(int)
		'''
		new_sent = []
		for bert_id in sent:
			new_id = self.bertid2id[bert_id]
			if new_id >= self.vocab_size and not invalid_vocab:
				new_id = self.unk_id
			new_sent.append(new_id)
		return new_sent

	def convert_tokens_to_ids(self, sent, invalid_vocab=False):
		'''Convert list of token(str) to list of id(int)

		Arguments:
				sent (list): list of token(str)

		Returns:
				ids (list): list of id(int)
		'''
		return self.sen_to_index(sent, invalid_vocab=invalid_vocab)

	def convert_ids_to_tokens(self, index, trim=True):
		'''Convert list of id(int) to list of token(str)

		Arguments:
				index (list): list of id(int)

		Returns:
				sent (list): list of token(str)
		'''
		return self.index_to_sen(index, trim=trim)

	def convert_ids_to_bert_ids(self, index):
		'''Convert list of id(int) to list of bert id(int)

		Arguments:
				index (list): list of id(int)

		Returns:
				bert_ids (list): list of bert id(int)
		'''
		return list(map(lambda word: self.id2bertid[word], index))

	def convert_bert_ids_to_tokens(self, sent, trim=True):
		'''Convert list of bert id(int) to list of token(str)

		Arguments:
				sent (list): list of bert id(int)

		Returns:
				(list): list of token(str)
		'''
		if trim:
			sent = trim_before_target(list(sent), self.bert_eos_id)
			idx = len(sent)
			while idx > 0 and sent[idx-1] == self.bert_pad_id:
				idx -= 1
			sent = sent[:idx]
		return list(map(lambda word: self.bert_id2word[word], sent))

	def get_batch(self, key, index):
		'''Get a batch of specified `index`.

		Arguments:
			key (str): must be contained in `key_name`
			index (list): a list of specified index

		Returns:
			(dict): A dict at least contains:

				* post_length (:class:`numpy.array`): A 1-d array, the length of post in each batch.
			  	  Size: `[batch_size]`
				* post (:class:`numpy.array`): A 2-d padding array containing id of words in posts.
			  	  Only provide valid words. `unk_id` will be used if a word is not valid.
			  	  Size: `[batch_size, max(sent_length)]`
				* post_allvocabs (:class:`numpy.array`): A 2-d padding array containing id of words in posts.
			  	  Provide both valid and invalid vocabs.
			  	  Size: `[batch_size, max(sent_length)]`
			  	* post_bert (:class:`numpy.array`): A 2-d padding array containing BERT id of words in posts.
			  	  Size: `[batch_size, max(sent_length)]`
				* resp_length (:class:`numpy.array`): A 1-d array, the length of response in each batch.
			  	  Size: `[batch_size]`
				* resp (:class:`numpy.array`): A 2-d padding array containing id of words in responses.
			  	  Only provide valid vocabs. `unk_id` will be used if a word is not valid.
			  	  Size: `[batch_size, max(sent_length)]`
				* resp_allvocabs (:class:`numpy.array`):
				  A 2-d padding array containing id of words in responses.
			  	  Provide both valid and invalid vocabs.
			  	  Size: `[batch_size, max(sent_length)]`
			  	* resp_bert (:class:`numpy.array`):
			  	  A 2-d padding array containing BERT id of words in responses.
			  	  Size: `[batch_size, max(sent_length)]`
		'''
		if key not in self.key_name:
			raise ValueError("No set named %s." % key)
		res = {}
		batch_size = len(index)
		res["post_length"] = np.array(list(map(lambda i: len(self.data[key]['post'][i]), index)))
		res["resp_length"] = np.array(list(map(lambda i: len(self.data[key]['resp'][i]), index)))
		res_post = res["post"] = np.zeros((batch_size, np.max(res["post_length"])), dtype=int)
		res_resp = res["resp"] = np.zeros((batch_size, np.max(res["resp_length"])), dtype=int)
		res_post_bert = res["post_bert"] = np.zeros((batch_size, np.max(res["post_length"])), dtype=int)
		res_resp_bert = res["resp_bert"] = np.zeros((batch_size, np.max(res["resp_length"])), dtype=int)
		for i, j in enumerate(index):
			post = self.data[key]['post'][j]
			resp = self.data[key]['resp'][j]
			res_post[i, :len(post)] = post
			res_resp[i, :len(resp)] = resp

			post_bert = self.data[key]['post_bert'][j]
			resp_bert = self.data[key]['resp_bert'][j]
			res_post_bert[i, :len(post_bert)] = post_bert
			res_resp_bert[i, :len(resp_bert)] = resp_bert

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
		metric.add_metric(BleuCorpusMetric(self, gen_key=gen_key))
		metric.add_metric(SingleTurnDialogRecorder(self, gen_key=gen_key))
		return metric
