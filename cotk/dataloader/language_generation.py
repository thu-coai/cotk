"""Dataloader for language generation"""
import numpy as np

# from .._utils.unordered_hash import UnorderedSha256
from .._utils.file_utils import get_resource_file_path
from .._utils import hooks
from .dataloader import LanguageProcessingBase
from ..metric import MetricChain, PerplexityMetric, LanguageGenerationRecorder, \
	FwBwBleuCorpusMetric, SelfBleuCorpusMetric

# pylint: disable=W0223
class LanguageGeneration(LanguageProcessingBase):
	r"""Base class for language modelling datasets. This is an abstract class.

	This class is supported for language modeling tasks or language generation tasks
	without any inputs.

	Arguments:{ARGUMENTS}

	Attributes:{ATTRIBUTES}
	"""

	ARGUMENTS = r'''
			file_id (str): A string indicating the source of language generation dataset. {FILE_ID_DEFAULT}
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

	_version = 1

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
		return super()._general_load_data(self._file_path, [['sent', 'Sentence']], \
			self._min_vocab_times, self._max_sent_length, None, self._invalid_vocab_times)

	def get_batch(self, key, indexes):
		'''{LanguageProcessingBase.GET_BATCH_DOC_WITHOUT_RETURNS}

		Returns:
			(dict): A dict at least contains:

			* **sent_length** (:class:`numpy.ndarray`): A 1-d array, the length of sentence in each batch.
			  Size: ``[batch_size]``
			* **sent** (:class:`numpy.ndarray`): A 2-d padding array containing id of words.
			  Only provide valid words. ``unk_id`` will be used if a word is not valid.
			  Size: ``[batch_size, max(sent_length)]``
			* **sent_allvocabs** (:class:`numpy.ndarray`): A 2-d padding array containing id of words.
			  Provide both valid and invalid words.
			  Size: ``[batch_size, max(sent_length)]``

		Examples:

			>>> # all_vocab_list = ["<pad>", "<unk>", "<go>", "<eos>", "how", "are", "you",
			>>> #	"hello", "i", "am", "fine"]
			>>> # vocab_size = 9
			>>> # vocab_list = ["<pad>", "<unk>", "<go>", "<eos>", "how", "are", "you", "hello", "i"]
			>>> dataloader.get_batch('train', [0, 1, 2])
			{
				"sent": numpy.array([
					[2, 4, 5, 6, 3, 0],   # first sentence: <go> how are you <eos> <pad>
					[2, 7, 3, 0, 0, 0],   # second sentence:  <go> hello <eos> <pad> <pad> <pad>
					[2, 7, 8, 1, 1, 3]    # third sentence: <go> hello i <unk> <unk> <eos>
				]),
				"sent_length": numpy.array([5, 3, 6]), # length of sentences
				"sent_allvocabs": numpy.array([
					[2, 4, 5, 6, 3, 0],   # first sentence: <go> how are you <eos> <pad>
					[2, 7, 3, 0, 0, 0],   # second sentence:  <go> hello <eos> <pad> <pad> <pad>
					[2, 7, 8, 9, 10, 3]   # third sentence: <go> hello i am fine <eos>
				]),
			}
		'''
		if key not in self.key_name:
			raise ValueError("No set named %s." % key)
		res = {}
		batch_size = len(indexes)
		res["sent_length"] = np.array( \
			list(map(lambda i: len(self.data[key]['sent'][i]), indexes)), dtype=int)
		res_sent = res["sent"] = np.zeros( \
			(batch_size, np.max(res["sent_length"])), dtype=int)
		for i, j in enumerate(indexes):
			sentence = self.data[key]['sent'][j]
			res["sent"][i, :len(sentence)] = sentence

		res["sent_allvocabs"] = res_sent.copy()
		res_sent[res_sent >= self.valid_vocab_len] = self.unk_id
		return res

	def get_teacher_forcing_metric(self, gen_log_prob_key="gen_log_prob"):
		'''Get metrics for teacher-forcing. In other words, this function
		provides metrics for language modelling task.

		It contains:

		* :class:`.metric.PerplexityMetric`

		Arguments:
			gen_log_prob_key (str): The key of predicted log probability over words.
				Refer to :class:`.metric.PerplexityMetric`. Default: ``gen_log_prob``.

		Returns:
			A :class:`.metric.MetricChain` object.
		'''
		metric = MetricChain()
		metric.add_metric(PerplexityMetric(self, \
					reference_allvocabs_key='sent_allvocabs', \
					reference_len_key='sent_length', \
					gen_log_prob_key=gen_log_prob_key))
		return metric

	def get_inference_metric(self, gen_key="gen", sample=1000, seed=1229, cpu_count=None):
		'''Get metrics for inference. In other words, this function provides metrics for
		language generation tasks.

		It contains:

		* :class:`.metric.SelfBleuCorpusMetric`
		* :class:`.metric.FwBwBleuCorpusMetric`
		* :class:`.metric.LanguageGenerationRecorder`

		Arguments:
			gen_key (str): The key of generated sentences in index form.
				Refer to :class:`.metric.LanguageGenerationRecorder`.
				Default: ``gen``.
			sample (int): Sample numbers for self-bleu metric.
				It will be fast but inaccurate if this become small.
				Refer to :class:`.metric.SelfBleuCorpusMetric`. Default: ``1000``.
			seed (int): Random seed for sampling.
				Refer to :class:`.metric.SelfBleuCorpusMetric`. Default: ``1229``.
			cpu_count (int): Number of used cpu for multiprocessing.
				Refer to :class:`.metric.SelfBleuCorpusMetric`. Default: ``None``.
		Returns:
			A :class:`.metric.MetricChain` object.
		'''
		metric = MetricChain()
		metric.add_metric(SelfBleuCorpusMetric(self, \
					gen_key=gen_key, \
					sample=sample, \
					seed=seed, \
					cpu_count=cpu_count))
		metric.add_metric(FwBwBleuCorpusMetric(self, \
					reference_test_list=self.get_all_batch("test")["sent"], \
					gen_key=gen_key, \
					sample=sample, \
					seed=seed, \
					cpu_count=cpu_count))
		metric.add_metric(LanguageGenerationRecorder(self, gen_key=gen_key))
		return metric

class MSCOCO(LanguageGeneration):
	'''A dataloader for preprocessed MSCOCO dataset.

	Arguments:{ARGUMENTS}

	Refer to :class:`.LanguageGeneration` for attributes and methods.

	References:
		[1] http://images.cocodataset.org/annotations/annotations_trainval2017.zip

		[2] Chen X, Fang H, Lin T Y, et al. Microsoft COCO Captions:
		Data Collection and Evaluation Server. arXiv:1504.00325, 2015.
	'''

	ARGUMENTS = LanguageGeneration.ARGUMENTS
	FILE_ID_DEFAULT = r'''Default: ``resources://MSCOCO``.'''
	VALID_VOCAB_TIMES_DEFAULT = r'''Default: ``10``.'''
	MAX_SENT_LENGTH = r'''Default: ``50``.'''
	INVALID_VOCAB_TIMES_DEFAULT = r'''Default: ``0`` (No unknown words).'''
	TOKENIZER_DEFAULT = r'''Default: ``nltk``'''
	REMAINS_CAPITAL_DEFAULT = r'''Default: ``True``'''

	def __init__(self, file_id="resources://MSCOCO", min_vocab_times=10, \
			max_sent_length=50, invalid_vocab_times=0, \
			tokenizer="nltk", remains_capital=True, \
			):
		super().__init__(file_id, min_vocab_times, max_sent_length, invalid_vocab_times, \
			tokenizer, remains_capital)
