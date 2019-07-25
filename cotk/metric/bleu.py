r"""
Containing some classes and functions about bleu evaluating results of models.
"""
import random
import os
import multiprocessing
from multiprocessing import Pool
import numpy as np
import tqdm
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
from .metric import MetricBase


def _replace_unk(_input, _unk_id, _target=-1):
	r'''Auxiliary function for replacing the unknown words:

	Arguments:
		_input (list): the references or hypothesis.
		_unk_id (int): id for unknown words.
		_target: the target word index used to replace the unknown words.

	Returns:

		* list: processed result.
	'''
	output = []
	for _list in _input:
		_output = []
		for ele in _list:
			_output.append(_target if ele == _unk_id else ele)
		output.append(_output)
	return output

def _sentence_bleu(ele):
	'''Auxiliary function for computing sentence bleu:

	Arguments:
		ele (tuple): A tuple (`reference sentences`, `a hypothesis sentence`).

	Returns:

		* int: **sentence-bleu** value.
	'''
	return sentence_bleu(ele[0], ele[1], smoothing_function=SmoothingFunction().method1)

class BleuCorpusMetric(MetricBase):
	'''Metric for calculating BLEU.

	Arguments:
		{MetricBase.DATALOADER_ARGUMENTS}
		{MetricBase.REFERENCE_ALLVOCABS_KEY_ARGUMENTS}
		{MetricBase.GEN_KEY_ARGUMENTS}
	'''

	def __init__(self, dataloader, ignore_smoothing_error=False,\
			reference_allvocabs_key="ref_allvocabs", gen_key="gen"):
		super().__init__()
		self.dataloader = dataloader
		self.ignore_smoothing_error = ignore_smoothing_error
		self.reference_allvocabs_key = reference_allvocabs_key
		self.gen_key = gen_key
		self.refs = []
		self.hyps = []

	def forward(self, data):
		'''Processing a batch of data.

		Arguments:
			data (dict): A dict at least contains the following keys:

				{MetricBase.FORWARD_REFERENCE_ALLVOCABS_ARGUMENTS}
				{MetricBase.FORWARD_GEN_ARGUMENTS}

				Here is an example for data:
					>>> # all_vocab_list = ["<pad>", "<unk>", "<go>", "<eos>", "I", "have",
					>>> #   "been", "to", "China"]
					>>> data = {
					... 	reference_allvocabs_key: [[2,4,3], [2,5,6,3]]
					...		gen_key: [[4,5,3], [6,7,8,3]]
					... }
		'''
		super().forward(data)
		gen = data[self.gen_key]
		resp = data[self.reference_allvocabs_key]

		if not isinstance(gen, (np.ndarray, list)):
			raise TypeError("Unknown type for gen.")
		if not isinstance(resp, (np.ndarray, list)):
			raise TypeError("Unknown type for resp")

		if len(resp) != len(gen):
			raise ValueError("Batch num is not matched.")

		relevant_data = []
		for gen_sen, resp_sen in zip(gen, resp):
			self.hyps.append(self.dataloader.trim(gen_sen))
			reference = list(self.dataloader.trim(resp_sen[1:]))
			relevant_data.append(reference)
			self.refs.append([reference])
		self._hash_relevant_data(relevant_data)

	def close(self):
		'''
		Returns:
			(dict): Return a dict which contains

			* **bleu**: bleu value.
			* **bleu hashvalue**: hash value for bleu metric, same hash value stands
			  for same evaluation settings.
		'''
		result = super().close()
		if (not self.hyps) or (not self.refs):
			raise RuntimeError("The metric has not been forwarded data correctly.")

		self.hyps = _replace_unk(self.hyps, self.dataloader.unk_id)
		try:
			result.update({"bleu": \
				corpus_bleu(self.refs, self.hyps, smoothing_function=SmoothingFunction().method3), \
				"bleu hashvalue": self._hashvalue()})
		except ZeroDivisionError as _:
			if not self.ignore_smoothing_error:
				raise ZeroDivisionError("Bleu smoothing divided by zero. This is a known bug of corpus_bleu, \
				usually caused when there is only one sample and the sample length is 1.")
			result.update({"bleu": \
					0, \
					"bleu hashvalue": self._hashvalue()})
		return result

class SelfBleuCorpusMetric(MetricBase):
	r'''Metric for calculating Self-BLEU.

	Arguments:
		{MetricBase.DATALOADER_ARGUMENTS}
		{MetricBase.GEN_KEY_ARGUMENTS}
		sample (int): Number of examples sampled from the generated sentences. Default: ``1000``.
		seed (int): Random seed for sampling. Default: ``1229``.
		{MetricBase.CPU_COUNT_ARGUMENTS}

	Warning:
		the calculation of ``hashvalue`` considers the actual sample size of hypotheses which
			will be less than ``sample`` if the size of hypotheses is smaller than ``sample``
	'''

	def __init__(self, dataloader, \
		gen_key="gen", \
		sample=1000, \
		seed=1229, \
		cpu_count=None):
		super().__init__()
		self.dataloader = dataloader
		self.gen_key = gen_key
		self.sample = sample
		self.hyps = []
		self.seed = seed
		if cpu_count is not None:
			self.cpu_count = cpu_count
		elif "CPU_COUNT" in os.environ and os.environ["CPU_COUNT"] is not None:
			self.cpu_count = int(os.environ["CPU_COUNT"])
		else:
			self.cpu_count = multiprocessing.cpu_count()

	def forward(self, data):
		'''Processing a batch of data.

		Arguments:
			data (dict): A dict at least contains the following keys:

				{MetricBase.FORWARD_GEN_ARGUMENTS}

				Here is an example for data:
					>>> # all_vocab_list = ["<pad>", "<unk>", "<go>", "<eos>", "I", "have",
					>>> #   "been", "to", "China"]
					>>> data = {
					...		gen_key: [[4,5,3], [6,7,8,3]]
					... }
		'''
		super().forward(data)
		gen = data[self.gen_key]

		if not isinstance(gen, (np.ndarray, list)):
			raise TypeError("Unknown type for gen.")

		for gen_sen in gen:
			self.hyps.append(self.dataloader.trim(gen_sen))

	def close(self):
		'''
		Returns:
			(dict): Return a dict which contains

			* **self-bleu**: self-bleu value.
		'''
		res = super().close()
		if not self.hyps:
			raise RuntimeError("The metric has not been forwarded data correctly.")
		if len(self.hyps) == 1:
			raise RuntimeError("Self-Bleu can't be computed because there is only 1 generated sentence.")

		if self.sample > len(self.hyps):
			self.sample = len(self.hyps)
		random.seed(self.seed)
		random.shuffle(self.hyps)
		ref = self.hyps[:self.sample]
		_ref = _replace_unk(ref, self.dataloader.unk_id)

		bleu_irl = []
		if self.sample >= 1000 and self.cpu_count > 1:
			tasks = ((ref[:i]+ref[i+1:self.sample], _ref[i]) for i in range(self.sample))
			pool = Pool(self.cpu_count)
			for ans in tqdm.tqdm(pool.imap_unordered( \
				_sentence_bleu, tasks, chunksize=20), total=self.sample):
				bleu_irl.append(ans)
			pool.close()
			pool.join()
		elif self.sample > 1:
			for i in range(self.sample):
				bleu_irl.append(_sentence_bleu((ref[:i]+ref[i+1:], _ref[i])))
		self._hash_relevant_data([self.seed, self.sample])
		res.update({"self-bleu" : 1.0 * sum(bleu_irl) / len(bleu_irl),\
					"self-bleu hashvalue": self._hashvalue()})
		return res

class FwBwBleuCorpusMetric(MetricBase):
	r'''Metric for calculating FwBw-BLEU.

	Arguments:
		{MetricBase.DATALOADER_ARGUMENTS}
		reference_test_key (str): Reference sentences with :ref:`all vocabs <vocab_ref>` in test data
			are passed to :func:`forward` by ``dataloader.data["test"][self.reference_test_key]``.
		{MetricBase.GEN_KEY_ARGUMENTS}
		sample (int): Number of examples sampled from the generated sentences. Default: ``1000``.
		seed (int): random seed for sampling. Default: ``1229``.
		{MetricBase.CPU_COUNT_ARGUMENTS}
	Warning:
		The calculation of ``hashvalue`` considers the actual sample size of hypotheses and
		references. Therefore ``hashvalue`` may vary with the size of hypothesis or references
		if the size of them is smaller than ``sample``.
	'''

	def __init__(self, dataloader, \
			reference_test_key, \
			gen_key="gen", \
			sample=1000, \
			seed=1229, \
			cpu_count=None):
		super().__init__()
		self.dataloader = dataloader
		self.reference_test_key = reference_test_key
		self.gen_key = gen_key
		self.sample = sample
		self.seed = seed
		if cpu_count is not None:
			self.cpu_count = cpu_count
		elif "CPU_COUNT" in os.environ and os.environ["CPU_COUNT"] is not None:
			self.cpu_count = int(os.environ["CPU_COUNT"])
		else:
			self.cpu_count = multiprocessing.cpu_count()
		self.refs = []
		self.hyps = []

	def forward(self, data):
		'''Processing a batch of data.

		Arguments:
			data (dict): A dict at least contains the following keys:

				{MetricBase.FORWARD_GEN_ARGUMENTS}

				Here is an example for data:
					>>> # all_vocab_list = ["<pad>", "<unk>", "<go>", "<eos>", "I", "have",
					>>> #   "been", "to", "China"]
					>>> data = {
					...		gen_key: [[4,5,3], [6,7,8,3]]
					... }
		'''
		gen = data[self.gen_key]

		if not isinstance(gen, (np.ndarray, list)):
			raise TypeError("Unknown type for gen.")

		for gen_sen in gen:
			self.hyps.append(list(self.dataloader.trim(gen_sen)))


	def close(self):
		'''
		Returns:
			(dict): Return a dict which contains

			* **fwbwbleu**: fw/bw bleu value.
			* **fw-bw-bleu hashvalue**: hash value for fwbwbleu metric, same hash value stands
			  for same evaluation settings.
		'''
		res = super().close()
		if not self.hyps:
			raise RuntimeError("The metric has not been forwarded data correctly.")

		resp = self.dataloader.data["test"][self.reference_test_key]
		for resp_sen in resp:
			self.refs.append(list(self.dataloader.trim(resp_sen[1:])))

		sample_hyps = self.sample if self.sample < len(self.hyps) else len(self.hyps)
		sample_refs = self.sample if self.sample < len(self.refs) else len(self.refs)

		random.seed(self.seed)
		random.shuffle(self.hyps)
		random.shuffle(self.refs)

		self.hyps = _replace_unk(self.hyps, self.dataloader.unk_id)

		bleu_irl_fw, bleu_irl_bw = [], []
		if sample_hyps >= 1000 and self.cpu_count > 1:
			tasks = ((self.refs, self.hyps[i]) for i in range(sample_hyps))
			pool = Pool(self.cpu_count)
			for ans in tqdm.tqdm(pool.imap_unordered( \
				_sentence_bleu, tasks, chunksize=20), total=sample_hyps):
				bleu_irl_fw.append(ans)
			pool.close()
			pool.join()
		else:
			for i in range(sample_hyps):
				bleu_irl_fw.append(_sentence_bleu((self.refs, self.hyps[i])))

		if sample_refs >= 1000 and self.cpu_count > 1:
			tasks = ((self.hyps, self.refs[i]) for i in range(sample_refs))
			pool = Pool(self.cpu_count)
			for ans in tqdm.tqdm(pool.imap_unordered( \
				_sentence_bleu, tasks, chunksize=20), total=sample_refs):
				bleu_irl_bw.append(ans)
			pool.close()
			pool.join()
		else:
			for i in range(sample_refs):
				bleu_irl_bw.append(_sentence_bleu((self.hyps, self.refs[i])))
		fw_bleu = (1.0 * sum(bleu_irl_fw) / len(bleu_irl_fw))
		bw_bleu = (1.0 * sum(bleu_irl_bw) / len(bleu_irl_bw))
		if fw_bleu + bw_bleu > 0:
			fw_bw_bleu = 2.0 * bw_bleu * fw_bleu / (fw_bleu + bw_bleu)
		else:
			fw_bw_bleu = 0

		res.update({"fw-bleu" : fw_bleu, \
			"bw-bleu" : bw_bleu, \
			"fw-bw-bleu" : fw_bw_bleu \
		})

		self._hash_relevant_data(self.refs + [self.seed, sample_hyps, sample_refs])
		res.update({"fw-bw-bleu hashvalue" : self._hashvalue()})
		return res

class MultiTurnBleuCorpusMetric(MetricBase):
	'''Metric for calculating multi-turn BLEU.

	Arguments:
		{MetricBase.DATALOADER_ARGUMENTS}
		{MetricBase.MULTI_TURN_REFERENCE_ALLVOCABS_KEY_ARGUMENTS}
		{MetricBase.MULTI_TURN_GEN_KEY_ARGUMENTS}
		{MetricBase.MULTI_TURN_LENGTH_KEY_ARGUMENTS}
	'''
	def __init__(self, dataloader, ignore_smoothing_error=False,\
					multi_turn_reference_allvocabs_key="reference_allvocabs", \
					multi_turn_gen_key="multi_turn_gen", \
					turn_len_key="turn_length" \
			  ):
		super().__init__()
		self.dataloader = dataloader
		self.ignore_smoothing_error = ignore_smoothing_error
		self.multi_turn_reference_allvocabs_key = multi_turn_reference_allvocabs_key
		self.turn_len_key = turn_len_key
		self.multi_turn_gen_key = multi_turn_gen_key
		self.refs = []
		self.hyps = []

	def forward(self, data):
		'''Processing a batch of data.

		Arguments:
			data (dict): A dict at least contains the following keys:

				{MetricBase.FORWARD_MULTI_TURN_REFERENCE_ALLVOCABS_ARGUMENTS}
				{MetricBase.FORWARD_MULTI_TURN_GEN_ARGUMENTS}
				{MetricBase.FORWARD_MULTI_TURN_LENGTH_ARGUMENTS}

				Here is an example for data:
					>>> # all_vocab_list = ["<pad>", "<unk>", "<go>", "<eos>", "I", "have",
					>>> #   "been", "to", "China"]
					>>> data = {
					...		multi_turn_reference_allvocabs_key: [[[2,4,3], [2,5,6,3]], [[2,7,6,8,3]]]
					...		turn_len_key: [2, 1]
					...		gen_key: [[[6,7,8,3], [4,5,3]], [[7,3]]]
					... }
		'''
		super().forward(data)
		reference_allvocabs = data[self.multi_turn_reference_allvocabs_key]
		length = data[self.turn_len_key]
		gen = data[self.multi_turn_gen_key]

		if not isinstance(reference_allvocabs, (np.ndarray, list)):
			raise TypeError("Unknown type for reference_allvocabs.")
		if not isinstance(length, (np.ndarray, list)):
			raise TypeError("Unknown type for length")
		if not isinstance(gen, (np.ndarray, list)):
			raise TypeError("Unknown type for gen")

		if len(length) != len(reference_allvocabs) or len(length) != len(gen):
			raise ValueError("Batch num is not matched.")

		for i, turn_length in enumerate(length):
			gen_session = gen[i]
			ref_session = reference_allvocabs[i]
			for j in range(turn_length):
				self.hyps.append(list(self.dataloader.trim(gen_session[j])))
				self.refs.append([list(self.dataloader.trim(ref_session[j])[1:])])

	def close(self):
		'''
		Returns:
			(dict): Return a dict which contains

			* **bleu**: bleu value.
			* **bleu hashvalue**: hash value for bleu metric, same hash value stands
			  for same evaluation settings.
		'''
		result = super().close()
		if (not self.hyps) or (not self.refs):
			raise RuntimeError("The metric has not been forwarded data correctly.")
		self.hyps = _replace_unk(self.hyps, self.dataloader.unk_id)

		self._hash_relevant_data(self.refs)

		try:
			result.update({"bleu": \
				corpus_bleu(self.refs, self.hyps, smoothing_function=SmoothingFunction().method3), \
				"bleu hashvalue": self._hashvalue()})
		except ZeroDivisionError as _:
			if not self.ignore_smoothing_error:
				raise ZeroDivisionError("Bleu smoothing divided by zero. This is a known bug of corpus_bleu, \
				usually caused when there is only one sample and the sample length is 1.")
			result.update({"bleu": \
					0, \
					"bleu hashvalue": self._hashvalue()})
		return result
