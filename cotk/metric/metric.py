"""
``cotk.metrics`` provides classes and functions evaluating results of models.
It provides a fair metric for every model.
"""
from typing import Any, List, Dict
import hashlib

from .._utils.unordered_hash import UnorderedSha256, dumps
from .._utils.metaclass import LoadClassInterface, DocStringInheritor

class MetricBase(LoadClassInterface, metaclass=DocStringInheritor):
	'''Base class for metrics.
	'''

	DATALOADER_ARGUMENTS = \
		"""dataloader (:class:`.dataloader.LanguageProcessing`, :class:`.dataloader.Sentence`, :class:`.dataloader.Session`): \
		 A language generation dataloader."""
	MULTI_TURN_DATALOADER_ARGUMENTS = \
		"""dataloader (:class:`.dataloader.LanguageProcessing`, :class:`.dataloader.Session`): \
		 A language generation dataloader."""
	NGRAM_ARGUMENTS = \
		"""ngram (int, optional): The order of ngram to calculate metrics like BLEU and Perplexity. Default: ``4``."""
	TOKENIZER_ARGUMENTS = \
		"""tokenizer (None, :class:`.dataloader.Tokenizer`, str, optional): Specifies the tokenizer used in \
			the metric. Default: ``None``."""
	IGNORE_SMOOTHING_ERROR_ARGUMENTS = \
		"""ignore_smoothing_error (bool, optional): Specifies whether to ignore the smoothing error when calculating \
			BLEU. Default: ``False``."""
	SAMPLE_ARGUMENTS_IN_BLEU = \
		"""sample (int, optional): Number of examples sampled from the generated sentences. Default: ``1000``."""
	SAMPLE_ARGUMENTS_IN_NGRAM_PERPLEXITY = \
		SAMPLE_ARGUMENTS_IN_BLEU.replace("Default: ``1000``.", "Default: ``10000``.")
	SEED_ARGUMENTS = \
		"""seed (int, optional): Random seed for sampling. Default: ``1229``."""
	REFERENCE_TEST_LIST_ARGUMENTS = \
		"""reference_test_list (list): Reference sentences with :ref:`all vocabs <vocabulary_ref>` in test data."""
	REFERENCE_ALLVOCABS_KEY_ARGUMENTS = \
		"""reference_allvocabs_key (str, optional): \
			The key of reference sentences. Default: ``ref_allvocabs``."""
	FORWARD_REFERENCE_ALLVOCABS_ARGUMENTS = \
				"""* **data[reference_allvocabs_key]** (list, :class:`numpy.ndarray`): \
				  A 2-d jagged or padded array of int. Reference sentences with \
				  :ref:`allvocabs <vocabulary_ref>` in index form. \
				  The sentences can optionally contain start tokens (eg: <go>), end tokens (eg: <eos>) and padding tokens (eg: <pad>), \
				  which will be removed in the recorder.
				  Size: ``[batch_size, ~ref_sentence_length]``, \
				  where "~" means different sizes in this dimension is allowed."""
	FORWARD_BLEU_REFERENCE_ALLVOCABS_ARGUMENTS = \
				"""* **data[reference_allvocabs_key]** (list, :class:`numpy.ndarray`): \
				  A 2-d (3-d) jagged or padded array of int. Reference sentences with \
				  :ref:`allvocabs <vocabulary_ref>` in index form. \
				  Contains start token (eg: ``<go>``) and end token (eg: ``<eos>``). \
				  Size: ``[batch_size, ~ref_sentence_length]`` (``[batch_size, ~ref_num, ~ref_sentence_length]``), \
				  where "~" means different sizes in this dimension is allowed.
				  Note that if this is a 3-d array, the second dim ``ref_num`` must be ``reference_num`` unless ``reference_num`` \
				  is explicitly set ``None``."""
	FORWARD_REFERENCE_ALLVOCABS_ARGUMENTS_WITH_TORCH = \
		FORWARD_REFERENCE_ALLVOCABS_ARGUMENTS.replace("list, :class:`numpy.ndarray`", \
			"list, :class:`numpy.ndarray`, :class:`torch.Tensor`")
	FORWARD_POST_ALLVOCABS_ARGUMENTS = \
		FORWARD_REFERENCE_ALLVOCABS_ARGUMENTS.replace("reference_allvocabs_key", \
			"post_allvocabs_key")
	FORWARD_RESP_ALLVOCABS_ARGUMENTS = \
		FORWARD_REFERENCE_ALLVOCABS_ARGUMENTS.replace("reference_allvocabs_key", \
			"resp_allvocabs_key")

	LABEL_KEY_ARGUMENTS = \
		"""label_key (str): \
			The key of reference sentence labels. Default: ``label``."""
	LABEL_ARGUMENTS = """* **data[label_key]** (list or :class:`numpy.ndarray`): \
				  A 1-d array of int. \
				  Size: ``[batch_size]``, \
				  each element refers to label of one sample"""

	PREDICTION_KEY_ARGUMENTS = \
		"""prediction_key (str): \
			The key of reference sentence predictions. Default: ``prediction``."""
	PREDICTION_ARGUMENTS = """* **data[prediction_key]** (list or :class:`numpy.ndarray`): \
				  A 1-d array of int. \
				  Size: ``[batch_size]``, \
				  each element refers to prediction for one sample"""

	MULTI_TURN_REFERENCE_ALLVOCABS_KEY_ARGUMENTS = \
		"""multi_turn_reference_allvocabs_key (str, optional): \
			The key of reference sentences. Default: ``multi_turn_ref_allvocabs``."""
	FORWARD_MULTI_TURN_REFERENCE_ALLVOCABS_ARGUMENTS = \
				"""* **data[multi_turn_reference_allvocabs_key]** (list, :class:`numpy.ndarray`): \
				  A 3-d jagged or padded array of int. Multi-turn reference sentences with \
				  :ref:`all vocabs <vocabulary_ref>`. \
				  Special tokens such as start token (eg: ``<go>``) and end token (eg: ``<eos>``) is optional (indifferent to the result). \
				  Padding token (i.e. ``<pad>``) is allowed.\
				  where "~" means different sizes in this dimension is allowed."""
	FORWARD_MULTI_TURN_REFERENCE_ALLVOCABS_ARGUMENTS_WITH_TORCH = \
		FORWARD_MULTI_TURN_REFERENCE_ALLVOCABS_ARGUMENTS.replace("list, :class:`numpy.ndarray`", \
			"list, :class:`numpy.ndarray`, :class:`torch.Tensor`")

	REFERENCE_LEN_KEY_ARGUMENTS = \
		"""reference_len_key (str, optional): \
			The key of lengths of reference sentences. \
			Default: ``ref_length``."""
	FORWARD_REFERENCE_LEN_ARGUMENTS = \
				"""* **data[reference_len_key]** (list, :class:`numpy.ndarray`): \
				  Length of reference sentences. Contains start token (eg:``<go>``) \
				  and end token (eg:``<eos>``). Size: ``[batch_size]``."""

	MULTI_TURN_REFERENCE_LEN_KEY_ARGUMENTS = \
		"""multi_turn_reference_len_key (str, optional): \
			The key of lengths of reference sentences. \
			Default: ``multi_turn_ref_length``."""
	FORWARD_MULTI_TURN_REFERENCE_LEN_ARGUMENTS = \
				"""* **data[multi_turn_reference_len_key]** (list, :class:`numpy.ndarray`): \
				  A 2-d jagged or padded array of int. **If padded, redundant position must be set to** ``0``. \
				  Length of multi-turn reference sentences. Contains start token (eg:``<go>``) \
				  and end token (eg:``<eos>``). Size: ``[batch_size, ~turn_length]``, \
				  where "~" means different sizes in this dimension is allowed."""

	GEN_KEY_ARGUMENTS = \
		"""gen_key (str, optional): \
			The key of generated sentences. Default: ``gen``."""
	GEN_LOG_PROB_KEY_ARGUMENTS = \
		"""gen_log_prob_key (str, optional): The key of predicted **log** probability over words. \
			Default: ``gen_log_prob``."""
	GENERATE_RARE_VOCAB_ARGUMENTS = \
		"""generate_rare_vocab (bool, optional): Whether ``gen_log_prob`` contains :ref:`invalid vocab <vocabulary_ref>`. \
			Default: ``False``."""
	FULL_CHECK_ARGUMENTS = \
		"""full_check (bool, optional): Whether to perform a full check on ``gen_log_prob`` to make sure the sum
			of probability is 1. Otherwise, a random check will be performed for efficiency.
			If PyTorch is used, a full check is always performed and this argument will be ignored.
			Default: ``False``."""
	FORWARD_GEN_ARGUMENTS = \
				"""* **data[gen_key]** (list, :class:`numpy.ndarray`): \
				  A 2-d jagged or padded array of int. \
				  Sentences generated by model. Special tokens such as start token \
				  The sentences can optionally contain start tokens (eg: <go>), end tokens (eg: <eos>) and padding tokens (eg: <pad>), \
				  which will be removed in the recorder.
				  Size: ``[batch_size, ~gen_sentence_length]``, \
				  where "~" means different sizes in this dimension is allowed."""

	MULTI_TURN_GEN_KEY_ARGUMENTS = \
		"""multi_turn_gen_key (str, optional): \
			The key of generated sentences. Default: ``multi_turn_gen``."""
	FORWARD_MULTI_TURN_GEN_ARGUMENTS = \
				"""* **data[gen_key]** (list, :class:`numpy.ndarray`): \
				  A 3-d jagged or padded array of int. Sentences generated by model. \
				  The sentences can optionally contain start tokens (eg: <go>), end tokens (eg: <eos>) and padding tokens (eg: <pad>), \
				  which will be removed in the recorder.
				  Size: ``[batch_size, ~max_turn_length, ~gen_sentence_length]``, \
				  where "~" means different sizes in this dimension is allowed."""

	MULTI_TURN_LENGTH_KEY_ARGUMENTS = \
		"""turn_length (str, optional): \
			The key of length of turns. Default: ``turn_length``."""
	FORWARD_MULTI_TURN_LENGTH_ARGUMENTS = \
				"""* **data[turn_len_key]** (list, :class:`numpy.ndarray`): \
				  Length of turns in each sample. \
				  Size: ``[batch_size]``."""

	CPU_COUNT_ARGUMENTS = \
		"""cpu_count (int, optional): Number of used cpu for multiprocessing. Multiprocessing will **NOT** be used \
			when ``cpu_count`` is set to ``1`` or the dataset is small. Default: If ``None``, \
			the environment variable ``CPU_COUNT`` will be used when available, \
			or all available cpu will be used otherwise."""

	def __init__(self, name: str, version: int):
		self.unordered_hash = UnorderedSha256()
		self.ordered_hash = hashlib.sha256()
		self.name = name
		self.version = version
		self._hash_ordered_data((name, version))
		self.closed = False

	def _hash_unordered_list(self, data_list: List[Any]):
		'''Invoked by :meth:`.forward` or :meth:`.close` to hash relevant data when computing a metric.

		Arguments:
			data_list (list): relevant data organized as list.
		'''
		for item in data_list:
			self.unordered_hash.update_data(dumps(item))

	def _hash_ordered_data(self, data: Any):
		self.ordered_hash.update(dumps(data))

	def _hashvalue(self):
		'''Invoked by :meth:`.close` to return the recorded hash value.
		'''
		return hashlib.sha256(dumps((self.ordered_hash.hexdigest(), self.unordered_hash.hexdigest()))).hexdigest()

	def forward(self, data: Dict[Any, Any]):
		'''Processing a batch of data.

		Arguments:
			data (dict): A dict contains the data that metrics need.
		'''
		if self.closed:
			raise ValueError("The metric has been closed.")
		if not isinstance(data, dict):
			raise TypeError("Data must be a dict.")

	def close(self) -> Dict[Any, Any]:
		'''
		Close the metric and return a dict containing results. Once the metric is closed,
		any operation on the metric (e.g. forward or another close) will raise a ValueError.
		'''
		if not self.closed:
			self.closed = True
			return {}
		else:
			raise RuntimeError("The metric has been closed.")

class MetricChain(MetricBase):
	'''A metric-like class for stacked metric. You can use this class
	making multiples metric combination like one.

	Examples:
		>>> metric = MetricChain()
		>>> metric.add_metric(BleuCorpusMetric())
		>>> metric.add_metric(SingleDialogRecorder(dataloader))

	Todo: Give more examples to combining forward and close
	'''
	_name = 'MetricChain'
	_version = 2
	def __init__(self):
		super().__init__(self._name, self._version)
		self.metric_list = []

	def add_metric(self, metric: "MetricBase"):
		'''Add metric for processing.

		Arguments:
			metric (:class:`.metric.MetricBase`): a metric class.
		'''
		if not isinstance(metric, MetricBase):
			raise TypeError("Metric must be a subclass of MetricBase")
		self.metric_list.append(metric)

	def forward(self, data: Dict[Any, Any]):
		'''Processing a batch of data.

		Arguments:
			data (dict): A dict at least contains keys which all the
				metric components need.
		'''
		super().forward(data)
		for metric in self.metric_list:
			metric.forward(data)

	def close(self) -> Dict[Any, Any]:
		r'''Return a dict containing the items which all the metric components return.
		'''
		res = super().close()
		for metric in self.metric_list:
			res.update(metric.close())
		return res
