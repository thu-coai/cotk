"""Dataloader for language generation"""
import numpy as np
from typing import List, Any, Tuple, Optional, Dict
from collections import OrderedDict

from ..hooks import hooks
from .dataloader import LanguageProcessing
from .context import FieldContext, VocabContext
from .tokenizer import PretrainedTokenizer
from .vocab import GeneralVocab, PretrainedVocab

if False: # for type check # pylint: disable=using-constant-test
	from ..metric import MetricChain #pylint: disable=unused-import

# pylint: disable=W0223
class LanguageGeneration(LanguageProcessing):
	"""Bases: :class:`.dataloader.LanguageProcessing`

	This class is supported for language modeling tasks or language generation tasks
	without any inputs.

	Arguments:{ARGUMENTS}
	"""

	ARGUMENTS = r'''
			file_id (str): A string indicating the source of language generation dataset. {FILE_ID_DEFAULT}
			min_frequent_vocab_times (int): A cut-off threshold of valid tokens. All tokens appear
				not less than ``min_frequent_vocab_times`` in **training set** will be marked as valid words.
				{MIN_FREQUENT_VOCAB_TIMES_DEFAULT}
			max_sent_length (int): All sentences longer than ``max_sent_length`` will be shortened
				to first ``max_sent_length`` tokens. {MAX_SENT_LENGTH}
			min_rare_vocab_times (int):  A cut-off threshold of invalid tokens. All tokens appear
				not less than ``min_rare_vocab_times`` in the **whole dataset** (except valid words) will be
				marked as invalid words. Otherwise, they are unknown words, which are ignored both for
				model or metrics. {MIN_RARE_VOCAB_TIMES_DEFAULT}
			tokenizer (str): How to tokenize sentence. ``nltk.tokenize.WordPunctTokenizer`` is used if ``nltk`` is specified,
				python built-in ``str.split`` is used if ``space`` is specified. {TOKENIZER_DEFAULT}
			convert_to_lower_letter(bool): Whether remaining capital letter in data or converting them to lower case. {CONVERT_TO_LOWER_LETTER_DEFAULT}
		'''
	FILE_ID_DEFAULT = ''
	MIN_FREQUENT_VOCAB_TIMES_DEFAULT = ''
	MAX_SENT_LENGTH = ''
	MIN_RARE_VOCAB_TIMES_DEFAULT = ''
	TOKENIZER_DEFAULT = ''
	CONVERT_TO_LOWER_LETTER_DEFAULT = ''

	_version = 2

	@hooks.hook_dataloader
	def __init__(self, file_id, *, tokenizer=None, \
			max_sent_length=None, \
			convert_to_lower_letter=None, \
			min_frequent_vocab_times=None, \
			min_rare_vocab_times=None, \
			pretrained=None):

		self._pretrained = pretrained
		if pretrained is None:
			with FieldContext.set_parameters(tokenizer=tokenizer,\
					max_sent_length=max_sent_length,
					convert_to_lower_letter=convert_to_lower_letter):
				with VocabContext.set_parameters(min_frequent_vocab_times=min_frequent_vocab_times, \
						min_rare_vocab_times=min_rare_vocab_times):
					super().__init__(file_id, OrderedDict([("sent", "SentenceDefault")]))
			self.set_default_field("train", "sent")

		elif pretrained == "gpt2":
			if not isinstance(tokenizer, PretrainedTokenizer):
				raise ValueError("tokenize should be loaded first if you want a gpt2 dataloader")
			vocab = PretrainedVocab(tokenizer.tokenizer)
			with FieldContext.set_parameters(tokenizer=tokenizer,\
					vocab=vocab, \
					max_sent_length=max_sent_length, \
					convert_to_lower_letter=convert_to_lower_letter):
				super().__init__(file_id, OrderedDict([("sent", "SentenceGPT2")]))
			self.set_default_field("train", "sent")
		else:
			raise ValueError("No pretrained name %s" % pretrained)

	_GET_BATCH_MORE_DOC = '''Returns a dict at least contains:

			* **sent_length** (:class:`numpy.ndarray`): A 1-d array, the length of sentence in each batch.
			  Size: ``[batch_size]``
			* **sent** (:class:`numpy.ndarray`): A 2-d padding array containing id of words.
			  Only provide valid words. ``unk_id`` will be used if a word is not valid.
			  Size: ``[batch_size, max(sent_length)]``
			* **sent_allvocabs** (:class:`numpy.ndarray`): A 2-d padding array containing id of words.
			  Provide both valid and invalid words.
			  Size: ``[batch_size, max(sent_length)]``'''

	_GET_BATCH_EXAMPLE = '''
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
			}'''

	def get_batch(self, set_name: str, indexes: List[int] #pylint: disable=useless-super-delegation
			) -> Dict[str, Any]:
		return super().get_batch(set_name, indexes)

	def get_teacher_forcing_metric(self, gen_log_prob_key="gen_log_prob") -> "MetricChain":
		'''Get metrics for teacher-forcing. In other words, this function
		provides metrics for language modelling task.

		It contains:

		* :class:`.metric.PerplexityMetric`

		Arguments:
			gen_log_prob_key (str): The key of predicted log probability over words.
				Refer to :class:`.metric.PerplexityMetric`. Default: ``gen_log_prob``.
		'''
		from ..metric import MetricChain, PerplexityMetric
		metric = MetricChain()
		metric.add_metric(PerplexityMetric(self, \
					reference_allvocabs_key='sent_allvocabs', \
					reference_len_key='sent_length', \
					gen_log_prob_key=gen_log_prob_key))
		return metric

	def get_inference_metric(self, gen_key="gen", sample=1000, seed=1229, cpu_count=None) -> "MetricChain":
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
		'''
		from ..metric import MetricChain, LanguageGenerationRecorder, \
			FwBwBleuCorpusMetric, SelfBleuCorpusMetric
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
	'''Bases: :class:`.dataloader.LanguageGeneration`

	A dataloader for preprocessed MSCOCO dataset.
	Refer to :class:`.LanguageGeneration` and :class:`.LanguageProcessing` for attributes and methods.

	Arguments:{ARGUMENTS}

	References:
		[1] http://images.cocodataset.org/annotations/annotations_trainval2017.zip

		[2] Chen X, Fang H, Lin T Y, et al. Microsoft COCO Captions:
		Data Collection and Evaluation Server. arXiv:1504.00325, 2015.
	'''

	ARGUMENTS = LanguageGeneration.ARGUMENTS
	FILE_ID_DEFAULT = r'''Default: ``resources://MSCOCO``.'''
	MIN_FREQUENT_VOCAB_TIMES_DEFAULT = r'''Default: ``10``.'''
	MAX_SENT_LENGTH = r'''Default: ``50``.'''
	MIN_RARE_VOCAB_TIMES_DEFAULT = r'''Default: ``0`` (No unknown words).'''
	TOKENIZER_DEFAULT = r'''Default: ``nltk``'''
	CONVERT_TO_LOWER_LETTER_DEFAULT = r'''Default: ``True``'''

	@hooks.hook_dataloader
	def __init__(self, file_id, *, tokenizer="nltk", \
			max_sent_length=50, \
			convert_to_lower_letter=False, \
			min_frequent_vocab_times=10, \
			min_rare_vocab_times=0, \
			pretrained=None):
		super().__init__(file_id, tokenizer=tokenizer, max_sent_length=max_sent_length,\
			convert_to_lower_letter=convert_to_lower_letter, min_frequent_vocab_times=min_frequent_vocab_times,\
			min_rare_vocab_times=min_rare_vocab_times, pretrained=pretrained)
