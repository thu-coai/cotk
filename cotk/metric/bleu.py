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
from .._utils import hooks


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

	Here is an exmaple:

		>>> dl = cotk.dataloader.UbuntuCorpus('resources://Ubuntu_small')
		>>> reference_allvocabs_key = "ref_allvocabs"
		>>> gen_key = "gen"
		>>> metric = cotk.metric.BleuCorpusMetric(dl,
		...	    reference_allvocabs_key=reference_allvocabs_key,
		...	    gen_key=gen_key)
		>>> data = {
		...     reference_allvocabs_key: [[2, 10, 64, 851, 3], [2, 10, 48, 851, 3]],
		...     # reference_allvocabs_key: [["<go>", "I", "like", "python", "<eos>"], ["<go>", "I", "use", "python", "<eos>"]],
		...
		...	    gen_key: [[10, 1028, 479, 285, 220, 3], [851, 17, 2451, 3]]
		...	    # gen_key: [["I", "love", "java", "very", "much", "<eos>"], ["python", "is", "excellent", "<eos>"]],
		... }
		>>> metric.forword(data)
		>>> metric.close()
		{'bleu': 0.08582363099612991,
		'bleu hashvalue': '70e019630fef24d9477034a3d941a5349fcbff5a3dc6978a13ea3d85290114fb'}

	'''

	_name = 'BleuCorpusMetric'
	_version = 1

	@hooks.hook_metric
	def __init__(self, dataloader, ignore_smoothing_error=False,\
			reference_allvocabs_key="ref_allvocabs", gen_key="gen"):
		super().__init__(self._name, self._version)
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
					...     reference_allvocabs_key: [[2,4,3], [2,5,6,3]],
					...	    gen_key: [[4,5,3], [6,7,8,3]]
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

	@hooks.hook_metric_close
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

	Here is an exmaple:

		>>> dl = cotk.dataloader.UbuntuCorpus('resources://Ubuntu_small')
		>>> gen_key = 'gen'
		>>> metric = cotk.metric.SelfBleuCorpusMetric(dl, gen_key=gen_key)
		>>> data = {
		...	    gen_key: [[10, 64, 851, 3], [10, 48, 851, 3]],
		...	    # gen_key: [["I", "like", "python", "<eos>"], ["I", "use", "python", "<eos>"]],
		... }
		>>> metric.forword(data)
		>>> metric.close()
		{'self-bleu': 0.13512001548070346,
		'self-bleu hashvalue': '53cf55829c1b080c86c392c846a5d39a54340c70d838ec953f952aa6731118fb'}
	'''

	_name = 'SelfBleuCorpusMetric'
	_version = 1

	@hooks.hook_metric
	def __init__(self, dataloader, \
		gen_key="gen", \
		sample=1000, \
		seed=1229, \
		cpu_count=None):
		super().__init__(self._name, self._version)
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
					...	    gen_key: [[4,5,3], [6,7,8,3]]
					... }
		'''
		super().forward(data)
		gen = data[self.gen_key]

		if not isinstance(gen, (np.ndarray, list)):
			raise TypeError("Unknown type for gen.")

		for gen_sen in gen:
			self.hyps.append(self.dataloader.trim(gen_sen))

	@hooks.hook_metric_close
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
		if self.sample <= 1:
			raise RuntimeError('`self.sample` should be more than 1, \
				whose value is `{}`'.format(self.sample))

		if self.sample > len(self.hyps):
			self.sample = len(self.hyps)

		rng_state = random.getstate()
		random.seed(self.seed)
		random.shuffle(self.hyps)
		random.setstate(rng_state)

		ref = self.hyps[:self.sample]
		_ref = _replace_unk(ref, self.dataloader.unk_id)

		bleu_irl = []

		tasks = ((ref[:i]+ref[i+1:self.sample], _ref[i]) for i in range(self.sample))
		if self.sample >= 1000 and self.cpu_count > 1:
			# use multiprocessing
			pool = Pool(self.cpu_count)
			values = pool.imap_unordered(_sentence_bleu, tasks, chunksize=20)
		else:
			pool = None
			values = map(_sentence_bleu, tasks)
		if self.sample >= 1000:
			# use tqdm
			values = tqdm.tqdm(values, total=self.sample)
		for ans in values:
			bleu_irl.append(ans)
		if pool is not None:
			pool.close()
			pool.join()

		self._hash_relevant_data([self.seed, self.sample])
		res.update({"self-bleu" : 1.0 * sum(bleu_irl) / len(bleu_irl),\
					"self-bleu hashvalue": self._hashvalue()})
		return res

class FwBwBleuCorpusMetric(MetricBase):
	r'''Metric for calculating FwBw-BLEU.

	Arguments:
		{MetricBase.DATALOADER_ARGUMENTS}
		reference_test_list (list): Reference sentences with :ref:`all vocabs <vocab_ref>` in test data.
		{MetricBase.GEN_KEY_ARGUMENTS}
		sample (int): Number of examples sampled from the generated sentences. Default: ``1000``.
		seed (int): random seed for sampling. Default: ``1229``.
		{MetricBase.CPU_COUNT_ARGUMENTS}
	Warning:
		The calculation of ``hashvalue`` considers the actual sample size of hypotheses and
		references. Therefore ``hashvalue`` may vary with the size of hypothesis or references
		if the size of them is smaller than ``sample``.

	Here is an exmaple:

		>>> dl = cotk.dataloader.UbuntuCorpus('resources://Ubuntu_small')
		>>> gen_key = 'gen'
		>>> metric = cotk.metric.FwBwBleuCorpusMetric(dl,
		...	    reference_test_list=dl.get_all_batch('test')['sent'][0],
		...	    gen_key=gen_key)
		>>> data = {
		...	    gen_key: [[10, 64, 851, 3], [10, 48, 851, 3]],
		...	    # gen_key: [["I", "like", "python", "<eos>"], ["I", "use", "python", "<eos>"]],
		... }
		>>> metric.forword(data)
		>>> metric.close()
		{'fw-bleu': 0.007688528488990184,
 		 'bw-bleu': 0.0012482612634667945,
 		 'fw-bw-bleu': 0.002147816509441494,
 		 'fw-bw-bleu hashvalue': '0e3f58a90225af615ff780f04c91613759e04a3c7b4329670b1d03b679adf8cd'}
	'''

	_name = 'FwBwBleuCorpusMetric'
	_version = 1

	@hooks.hook_metric
	def __init__(self, dataloader, \
			reference_test_list, \
			gen_key="gen", \
			sample=1000, \
			seed=1229, \
			cpu_count=None):
		super().__init__(self._name, self._version)
		self.dataloader = dataloader
		self.reference_test_list = reference_test_list
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
					...	    gen_key: [[4,5,3], [6,7,8,3]]
					... }
		'''
		gen = data[self.gen_key]

		if not isinstance(gen, (np.ndarray, list)):
			raise TypeError("Unknown type for gen.")

		for gen_sen in gen:
			self.hyps.append(list(self.dataloader.trim(gen_sen)))

	@hooks.hook_metric_close
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

		for resp_sen in self.reference_test_list:
			self.refs.append(list(self.dataloader.trim(resp_sen[1:])))

		sample_hyps = self.sample if self.sample < len(self.hyps) else len(self.hyps)
		sample_refs = self.sample if self.sample < len(self.refs) else len(self.refs)

		if sample_hyps <= 1:
			raise RuntimeError('`sample_hyps` should be more than 1, \
				whose value is `{}`'.format(sample_hyps))
		if sample_refs <= 1:
			raise RuntimeError('`sample_refs` should be more than 1, \
				whose value is `{}`'.format(sample_refs))

		rng_state = random.getstate()
		random.seed(self.seed)
		random.shuffle(self.hyps)
		random.shuffle(self.refs)
		random.setstate(rng_state)

		self.hyps = _replace_unk(self.hyps, self.dataloader.unk_id)


		bleu_irl_fw, bleu_irl_bw = [], []

		tasks = ((self.refs, self.hyps[i]) for i in range(sample_hyps))
		if sample_hyps >= 1000 and self.cpu_count > 1:
			pool = Pool(self.cpu_count)
			values = pool.imap_unordered(_sentence_bleu, tasks, chunksize=20)
		else:
			pool = None
			values = map(_sentence_bleu, tasks)
		if sample_hyps >= 1000:
			values = tqdm.tqdm(values, total=sample_hyps)
		for ans in values:
			bleu_irl_fw.append(ans)
		if pool is not None:
			pool.close()
			pool.join()

		tasks = ((self.hyps, self.refs[i]) for i in range(sample_refs))
		if sample_refs >= 1000 and self.cpu_count > 1:
			pool = Pool(self.cpu_count)
			values = pool.imap_unordered(_sentence_bleu, tasks, chunksize=20)
		else:
			pool = None
			values = map(_sentence_bleu, tasks)
		if sample_refs >= 1000:
			values = tqdm.tqdm(values, total=sample_refs)
		for ans in values:
			bleu_irl_bw.append(ans)
		if pool is not None:
			pool.close()
			pool.join()

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

	Here is an exmaple:

		>>> dl = cotk.dataloader.UbuntuCorpus('resources://Ubuntu_small')
		>>> multi_turn_reference_allvocabs_key = "reference_allvocabs"
		>>> multi_turn_gen_key = "multi_turn_gen"
		>>> turn_len_key = "turn_length"
		>>> metric = cotk.metric.MultiTurnBleuCorpusMetric(dl,
		>>>     multi_turn_reference_allvocabs_key=multi_turn_reference_allvocabs_key,
		>>>     multi_turn_gen_key=multi_turn_gen_key,
		>>>     turn_len_key=turn_len_key)
		>>> data = {
		...	    multi_turn_reference_allvocabs_key: [[[2, 10, 64, 851, 3], [2, 10, 64, 479, 3]], [[2, 10, 64, 279, 1460, 3]]],
		...     # multi_turn_reference_allvocabs_key = [[["<go>", "I", "like", "python", "<eos>"], ["<go>", "I", "like", "java", "<eos>"]],
		...     # 	[["<go>", "I", "like", "machine", "learning", "<eos>"]]]
		...
		...	    turn_len_key: [2, 1],
		...     # turn_len_key: [len(multi_turn_reference_allvocabs_key[0]), len(multi_turn_reference_allvocabs_key[1])]
		...
		...	    multi_turn_gen_key: [[[851, 17, 2451, 3], [2019, 17, 393, 3]], [[10, 64, 34058, 805, 2601, 3]]]
		...     # multi_turn_gen_key = [[["python", "is", "excellent", "<eos>"], ["PHP", "is", "best", "<eos>"]],
		...     # 	[["I", "like", "natural", "language", "processing", "<eos>"]]]
		... }
		>>> metric.forword(data)
		>>> metric.close()
		{'bleu': 0.12081744577265555,
		'bleu hashvalue': 'c65b44c454dee5a8d393901644c7f1acfdb847bae3ab03823cb5b9f643958960'}
	'''

	_name = 'MultiTurnBleuCorpusMetric'
	_version = 1

	@hooks.hook_metric
	def __init__(self, dataloader, ignore_smoothing_error=False,\
					multi_turn_reference_allvocabs_key="reference_allvocabs", \
					multi_turn_gen_key="multi_turn_gen", \
					turn_len_key="turn_length" \
			  ):
		super().__init__(self._name, self._version)
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
					...	    multi_turn_reference_allvocabs_key: [[[2,4,3], [2,5,6,3]], [[2,7,6,8,3]]],
					...	    turn_len_key: [2, 1],
					...	    gen_key: [[[6,7,8,3], [4,5,3]], [[7,3]]]
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

	@hooks.hook_metric_close
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
