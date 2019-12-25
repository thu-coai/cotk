'''
A module for single turn dialog.
'''
import os
import time
from collections import Counter
from itertools import chain
import multiprocessing
from multiprocessing import Pool
import tqdm

import numpy as np

from nltk.tokenize import WordPunctTokenizer
from .._utils.file_utils import get_resource_file_path
from .._utils import hooks
from .dataloader import LanguageProcessingBase
from .bert_dataloader import BERTLanguageProcessingBase
from ..metric import MetricChain, PerplexityMetric, BleuCorpusMetric, SingleTurnDialogRecorder

# pylint: disable=W0223
class SingleTurnDialog(LanguageProcessingBase):
	r"""Base class for single-turn dialog datasets. This is an abstract class.

	This class is supported for sequence to sequence generation tasks, especially
	single turn dialog tasks.

	Arguments:{ARGUMENTS}

	Attributes:{ATTRIBUTES}
	"""

	_version = 1

	ARGUMENTS = r'''
			file_id (str): A string indicating the source of single turn dialog dataset. {FILE_ID_DEFAULT}
			valid_vocab_times (int): A cut-off threshold of valid tokens. All tokens appear
				not less than ``min_vocab_times`` in **training set** will be marked as valid words.
				{VALID_VOCAB_TIMES_DEFAULT}
			max_sent_length (int): All sentences longer than ``max_sent_length`` will be shortened
				to first ``max_sent_length`` tokens. {MAX_SENT_LENGTH}
			invalid_vocab_times (int):  A cut-off threshold of invalid tokens. All tokens appear
				not less than ``invalid_vocab_times`` in the **whole dataset** (except valid words) will be
				marked as invalid words. Otherwise, they are unknown words, which are ignored both for
				model or metrics. {INVALID_VOCAB_TIMES_DEFAULT}
			tokenizer (str): How to tokenize sentence. ``nltk.tokenize.WordPunctTokenizer`` is used if ``nltk`` is specified,
				python built-in ``str.split`` is used if ``space`` is specified. {TOKENIZER_DEFAULT}
			remains_capital(bool): Whether remaining capital letter in data or converting them to lower case. {REMAINS_CAPITAL_DEFAULT}
		'''
	FILE_ID_DEFAULT = ''
	VALID_VOCAB_TIMES_DEFAULT = ''
	MAX_SENT_LENGTH = ''
	INVALID_VOCAB_TIMES_DEFAULT = ''
	TOKENIZER_DEFAULT = ''
	REMAINS_CAPITAL_DEFAULT = ''

	ATTRIBUTES = LanguageProcessingBase.ATTRIBUTES

	@hooks.hook_dataloader
	def __init__(self, file_id, min_vocab_times, \
			max_sent_length, invalid_vocab_times, \
			tokenizer, remains_capital, \
			):
		self._file_id = file_id
		self._file_path = get_resource_file_path(file_id)
		self._min_vocab_times = min_vocab_times
		self._max_sent_length = max_sent_length
		self._invalid_vocab_times = invalid_vocab_times
		self._tokenizer = tokenizer
		self._remains_capital = remains_capital
		super().__init__()

	def tokenize(self, sentence, remains_capital=None, tokenizer=None):
		r'''Convert sentence(str) to a list of tokens(str)

		Arguments:
			sentence (str): a string to be tokenized

		Returns:
			list: a list of tokens(str)
		'''
		return super().tokenize(sentence, remains_capital or self._remains_capital, \
			tokenizer or self._tokenizer)

	def _load_data(self):
		r'''Loading dataset, invoked during the initialization of :class:`LanguageProcessingBase`.
		'''
		return super()._general_load_data(self._file_path, [['post', 'Sentence'], ['resp', 'Sentence']], \
			self._min_vocab_times, self._max_sent_length, None, self._invalid_vocab_times)

	def get_batch(self, key, indexes):
		'''{LanguageProcessingBase.GET_BATCH_DOC_WITHOUT_RETURNS}

		Returns:
			(dict): A dict at least contains:

			* **post_length** (:class:`numpy.ndarray`): A 1-d array, the length of post in each batch.
			  Size: ``[batch_size]``
			* **post** (:class:`numpy.ndarray`): A 2-d padded array containing words of id form in posts.
			  Only provide valid words. ``unk_id`` will be used if a word is not valid.
			  Size: ``[batch_size, max(sent_length)]``
			* **post_allvocabs** (:class:`numpy.ndarray`): A 2-d padded array containing words of id
			  form in posts. Provide both valid and invalid vocabs.
			  Size: ``[batch_size, max(sent_length)]``
			* **resp_length** (:class:`numpy.ndarray`): A 1-d array, the length of response in each batch.
			  Size: ``[batch_size]``
			* **resp** (:class:`numpy.ndarray`): A 2-d padded array containing words of id form
			  in responses. Only provide valid vocabs. ``unk_id`` will be used if a word is not valid.
			  Size: ``[batch_size, max(sent_length)]``
			* **resp_allvocabs** (:class:`numpy.ndarray`):
			  A 2-d padded array containing words of id form in responses.
			  Provide both valid and invalid vocabs.
			  Size: ``[batch_size, max(sent_length)]``

		Examples:
			>>> # all_vocab_list = ["<pad>", "<unk>", "<go>", "<eos>", "how", "are", "you",
			>>> #	"hello", "i", "am", "fine"]
			>>> # vocab_size = 9
			>>> # vocab_list = ["<pad>", "<unk>", "<go>", "<eos>", "how", "are", "you", "hello", "i"]
			>>> dataloader.get_batch('train', [0, 1])
			{
				"post_allvocabs": numpy.array([
					[2, 5, 6, 10, 3],  # first post:  <go> are you fine <eos>
					[2, 7, 3, 0, 0],   # second post: <go> hello <eos> <pad> <pad>
				]),
				"post": numpy.array([
					[2, 5, 6, 1, 3],   # first post:  <go> are you <unk> <eos>
					[2, 7, 3, 0, 0],   # second post: <go> hello <eos> <pad> <pad>
				]),
				"resp_allvocabs": numpy.array([
					[2, 8, 9, 10, 3],  # first response:  <go> i am fine <eos>
					[2, 7, 3, 0, 0],   # second response: <go> hello <eos> <pad> <pad>
				]),
				"resp": numpy.array([
					[2, 8, 1, 1, 3],   # first response:  <go> i <unk> <unk> <eos>
					[2, 7, 3, 0, 0],   # second response: <go> hello <eos> <pad> <pad>
				]),
				"post_length": numpy.array([5, 3]), # length of posts
				"resp_length": numpy.array([5, 3]), # length of responses
			}
		'''
		if key not in self.key_name:
			raise ValueError("No set named %s." % key)
		res = {}
		batch_size = len(indexes)
		res["post_length"] = np.array(list(map(lambda i: len(self.data[key]['post'][i]), indexes)), dtype=int)
		res["resp_length"] = np.array(list(map(lambda i: len(self.data[key]['resp'][i]), indexes)), dtype=int)
		res_post = res["post"] = np.zeros((batch_size, np.max(res["post_length"])), dtype=int)
		res_resp = res["resp"] = np.zeros((batch_size, np.max(res["resp_length"])), dtype=int)
		for i, j in enumerate(indexes):
			post = self.data[key]['post'][j]
			resp = self.data[key]['resp'][j]
			res_post[i, :len(post)] = post
			res_resp[i, :len(resp)] = resp

		res["post_allvocabs"] = res_post.copy()
		res["resp_allvocabs"] = res_resp.copy()
		res_post[res_post >= self.valid_vocab_len] = self.unk_id
		res_resp[res_resp >= self.valid_vocab_len] = self.unk_id
		return res

	def get_teacher_forcing_metric(self, gen_log_prob_key="gen_log_prob",\
					   invalid_vocab=False):
		'''Get metrics for teacher-forcing.

		It contains:

		* :class:`.metric.PerplexityMetric`

		Arguments:
			gen_log_prob_key (str):  The key of predicted log probability over words.
				Refer to :class:`.metric.PerplexityMetric`. Default: ``gen_log_prob``.
			invalid_vocab (bool): Whether ``gen_log_prob`` contains invalid vocab.
				Refer to :class:`.metric.PerplexityMetric`. Default: ``False``.


		Returns:
			A :class:`.metric.MetricChain` object.
		'''
		metric = MetricChain()
		metric.add_metric(PerplexityMetric(self,\
			reference_allvocabs_key="resp_allvocabs",\
			reference_len_key="resp_length",\
			gen_log_prob_key=gen_log_prob_key,\
			invalid_vocab=invalid_vocab))
		return metric

	def get_inference_metric(self, gen_key="gen"):
		'''Get metrics for inference.

		It contains:

		* :class:`.metric.BleuCorpusMetric`
		* :class:`.metric.SingleTurnDialogRecorder`

		Arguments:
			gen_key (str): The key of generated sentences in index form.
				Refer to :class:`.metric.BleuCorpusMetric` or
				:class:`.metric.SingleTurnDialogRecorder`. Default: ``gen``.

		Returns:
			A :class:`.metric.MetricChain` object.
		'''
		metric = MetricChain()
		metric.add_metric(BleuCorpusMetric(self, gen_key=gen_key, \
			reference_allvocabs_key="resp_allvocabs"))
		metric.add_metric(SingleTurnDialogRecorder(self, gen_key=gen_key))
		return metric

class OpenSubtitles(SingleTurnDialog):
	'''A dataloader for OpenSubtitles dataset.

	Arguments:{ARGUMENTS}

	Refer to :class:`.SingleTurnDialog` for attributes and methods.

	References:
		[1] http://opus.nlpl.eu/OpenSubtitles.php

		[2] P. Lison and J. Tiedemann, OpenSubtitles2016: Extracting Large Parallel Corpora from
		Movie and TV Subtitles. LREC 2016.
	'''

	ARGUMENTS = SingleTurnDialog.ARGUMENTS
	FILE_ID_DEFAULT = r'''Default: ``resources://OpenSubtitles``.'''
	VALID_VOCAB_TIMES_DEFAULT = r'''Default: ``10``.'''
	MAX_SENT_LENGTH = r'''Default: ``50``.'''
	INVALID_VOCAB_TIMES_DEFAULT = r'''Default: ``0`` (No unknown words).'''
	TOKENIZER_DEFAULT = r'''Default: ``nltk``'''
	REMAINS_CAPITAL_DEFAULT = r'''Default: ``False``'''
	@hooks.hook_dataloader
	def __init__(self, file_id="resources://OpenSubtitles", min_vocab_times=10, \
			max_sent_length=50, invalid_vocab_times=0, \
			tokenizer="nltk", remains_capital=False\
			):
		super().__init__(file_id, min_vocab_times, max_sent_length, \
			invalid_vocab_times, tokenizer, remains_capital)

class BERTSingleTurnDialog(BERTLanguageProcessingBase):
	r"""Base class for single-turn dialog datasets **with BERT input**.
	This is an abstract class.

	This class is supported for sequence to sequence generation tasks, especially
	single turn dialog tasks.

	Arguments:{ARGUMENTS}

	Attributes:{ATTRIBUTES}
	"""

	ARGUMENTS = BERTLanguageProcessingBase.ARGUMENTS
	ATTRIBUTES = BERTLanguageProcessingBase.ATTRIBUTES

	def get_batch(self, key, indexes):
		'''{LanguageProcessingBase.GET_BATCH_DOC_WITHOUT_RETURNS}

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
		batch_size = len(indexes)
		res["post_length"] = np.array(list(map(lambda i: len(self.data[key]['post'][i]), indexes)), dtype=int)
		res["resp_length"] = np.array(list(map(lambda i: len(self.data[key]['resp'][i]), indexes)), dtype=int)
		res_post = res["post"] = np.zeros((batch_size, np.max(res["post_length"])), dtype=int)
		res_resp = res["resp"] = np.zeros((batch_size, np.max(res["resp_length"])), dtype=int)
		res_post_bert = res["post_bert"] = np.zeros((batch_size, np.max(res["post_length"])), dtype=int)
		res_resp_bert = res["resp_bert"] = np.zeros((batch_size, np.max(res["resp_length"])), dtype=int)
		for i, j in enumerate(indexes):
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

	def get_teacher_forcing_metric(self, gen_log_prob_key="gen_log_prob",\
					   invalid_vocab=False):
		'''Get metrics for teacher-forcing.

		It contains:

		* :class:`.metric.PerplexityMetric`

		Arguments:
			gen_log_prob_key (str):  The key of predicted log probablilty over words.
				Refer to :class:`.metric.PerplexityMetric`. Default: ``gen_log_prob``.
			invalid_vocab (bool): Whether ``gen_log_prob`` contains invalid vocab.
				Refer to :class:`.metric.PerplexityMetric`. Default: ``False``.

		Returns:
			A :class:`.metric.MetricChain` object.
		'''
		metric = MetricChain()
		metric.add_metric(PerplexityMetric(self,\
			reference_allvocabs_key="resp_allvocabs",\
			reference_len_key="resp_length",\
			gen_log_prob_key=gen_log_prob_key,\
			invalid_vocab=invalid_vocab))
		return metric

	def get_inference_metric(self, gen_key="gen"):
		'''Get metrics for inference.

		It contains:

		* :class:`.metric.BleuCorpusMetric`
		* :class:`.metric.SingleTurnDialogRecorder`

		Arguments:
			gen_key (str): The key of generated sentences in index form.
				Refer to :class:`.metric.BleuCorpusMetric` or
				:class:`.metric.SingleTurnDialogRecorder`. Default: ``gen``.
		Returns:
			A :class:`.metric.MetricChain` object.
		'''
		metric = MetricChain()
		metric.add_metric(BleuCorpusMetric(self, gen_key=gen_key, \
			reference_allvocabs_key="resp_allvocabs"))
		metric.add_metric(SingleTurnDialogRecorder(self, gen_key=gen_key))
		return metric


class BERTOpenSubtitles(BERTSingleTurnDialog):
	'''A dataloader for OpenSubtitles dataset.

	Arguments:
		file_id (str): a str indicates the source of OpenSubtitles dataset.
		min_vocab_times (int): A cut-off threshold of `UNK` tokens. All tokens appear
			less than `min_vocab_times`	will be replaced by `<unk>`. Default: 10.
		max_sent_length (int): All sentences longer than `max_sent_length` will be shortened
			to first `max_sent_length` tokens. Default: 50.
		invalid_vocab_times (int):  A cut-off threshold of invalid tokens. All tokens appear
			not less than `invalid_vocab_times` in the **whole dataset** (except valid words) will be
			marked as invalid vocabs. Otherwise, they are unknown words, both in training or
			testing stages. Default: 0 (No unknown words).
		bert_vocab (str): The vocab file of BERT used for this task. It should be a bert model name
							or local path. Default: `bert-base-uncased`.

	Refer to :class:`.BERTLanguageProcessingBase` for attributes and methods.

	References:
		[1] http://opus.nlpl.eu/OpenSubtitles.php

		[2] P. Lison and J. Tiedemann, OpenSubtitles2016: Extracting Large Parallel Corpora from
		Movie and TV Subtitles. LREC 2016.
	'''

	@hooks.hook_dataloader
	def __init__(self, file_id, min_vocab_times=10, \
			max_sent_length=50, invalid_vocab_times=0, \
			bert_vocab_name='bert-base-uncased', cpu_count=None):
		self._file_id = file_id
		self._file_path = get_resource_file_path(file_id)
		self._min_vocab_times = min_vocab_times
		self._max_sent_length = max_sent_length
		self._invalid_vocab_times = invalid_vocab_times

		if cpu_count is not None:
			self.cpu_count = cpu_count
		elif "CPU_COUNT" in os.environ and os.environ["CPU_COUNT"] is not None:
			self.cpu_count = int(os.environ["CPU_COUNT"])
		else:
			self.cpu_count = multiprocessing.cpu_count()

		super().__init__(bert_vocab_name=bert_vocab_name)

	@classmethod
	def _run_tokenize(cls, ele):
		def _tokenize(tokenizer, line):
			tokens = tokenizer.tokenize('[CLS] %s [SEP]' % (line))
			bert_ids = tokenizer.convert_tokens_to_ids(tokens)
			return tokens, bert_ids
		(post, resp) = ele
		post_tokens, post_bert_ids = _tokenize(cls.tokenizer, post)
		resp_tokens, resp_bert_ids = _tokenize(cls.tokenizer, resp)
		return post_tokens, post_bert_ids, resp_tokens, resp_bert_ids

	@classmethod
	def _set_tokenizer(cls, tokenizer):
		cls.tokenizer = tokenizer

	MP_SMALL_SIZE = 100
	def _mp_process(self, posts, resps):
		tasks = ((post, resp) for post, resp in zip(posts, resps))

		pool = None
		map_func = None
		if len(posts) < __class__.MP_SMALL_SIZE:
			self._set_tokenizer(self.tokenizer)
			map_func = map
		else:
			pool = Pool(self.cpu_count, \
				initializer=self._set_tokenizer, initargs=(self.tokenizer, ))
			map_func = lambda func, tasks: tqdm.tqdm(pool.imap_unordered(func, tasks, chunksize=500), \
				total=len(posts))
				
		post_tokens, post_bert_ids = [], []
		resp_tokens, resp_bert_ids = [], []
		
		for _post_tokens, _post_bert_ids, _resp_tokens, _resp_bert_ids in \
				map_func(self._run_tokenize, tasks):
			post_tokens.append(_post_tokens)
			post_bert_ids.append(_post_bert_ids[:self._max_sent_length])
			resp_tokens.append(_resp_tokens)
			resp_bert_ids.append(_resp_bert_ids[:self._max_sent_length])

		if pool is not None:
			pool.close()
			pool.join()

		return post_tokens, post_bert_ids, resp_tokens, resp_bert_ids

	def _load_data(self):
		r'''Loading dataset, invoked by `BERTLanguageProcessingBase.__init__`
		'''
		print('begin load data...')
		begin_time = time.time()
		origin_data = {}
		for key in self.key_name:
			f_file = open("%s/opensub_pair_%s.post" % (self._file_path, key), 'r', encoding='utf-8')
			g_file = open("%s/opensub_pair_%s.response" % (self._file_path, key), 'r', encoding='utf-8')
			post_tokens, post_bert_ids, resp_tokens, resp_bert_ids = \
						self._mp_process(f_file.readlines(), g_file.readlines())
			origin_data[key] = {}
			origin_data[key]['post'] = post_tokens
			origin_data[key]['resp'] = resp_tokens
			origin_data[key]['post_bert'] = post_bert_ids
			origin_data[key]['resp_bert'] = resp_bert_ids

		print('finish tokenizing sentences...%f' % (time.time() - begin_time))

		raw_vocab_list = list(chain(*(origin_data['train']['post'] + origin_data['train']['resp'])))
		# Important: Sort the words preventing the index changes between different runs
		vocab = sorted(Counter(raw_vocab_list).most_common(), key=lambda pair: (-pair[1], pair[0]))
		left_vocab = list(filter(lambda x: x[1] >= self._min_vocab_times, vocab))
		vocab_list = self.ext_vocab + [x[0] for x in left_vocab if x[0] not in self.ext_vocab]
		valid_vocab_len = len(vocab_list)
		valid_vocab_set = set(vocab_list)

		left_vocab = list(filter(lambda x: x not in valid_vocab_set, self.bert_id2word))
		vocab_list.extend(left_vocab)

		print("valid vocab list length = %d" % valid_vocab_len)
		print("vocab list length = %d" % len(vocab_list))

		word2id = {w: i for i, w in enumerate(vocab_list)}
		line2id = lambda line: ( \
						list(map(lambda word: word2id[word], line)) \
					)[:self._max_sent_length]

		data = {}
		data_size = {}
		for key in self.key_name:
			data[key] = {}
			data[key]['post_bert'] = origin_data[key]['post_bert']
			data[key]['resp_bert'] = origin_data[key]['resp_bert']

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
			cut_num = np.sum(np.maximum(np.array(length) - self._max_sent_length + 1, 0))
			print("%s set. invalid rate: %f, unknown rate: %f, max length before cut: %d, \
					cut word rate: %f" % \
					(key, invalid_num / vocab_num, oov_num / vocab_num, max(length), cut_num / vocab_num))
		return vocab_list, valid_vocab_len, data, data_size
