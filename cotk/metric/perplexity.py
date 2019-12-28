r"""
Containing some classes and functions about perplexity evaluating results of models.
"""
import random

import numpy as np
from .metric import MetricBase
from .._utils.imports import LazyObject, LazyModule
from .._utils import hooks

torch = LazyModule("torch", globals())
torch.Tensor = LazyObject("torch.Tensor")

class PerplexityMetric(MetricBase):
	'''Metric for calculating perplexity.

	Arguments:
		{MetricBase.DATALOADER_ARGUMENTS}
		{MetricBase.REFERENCE_ALLVOCABS_KEY_ARGUMENTS}
		{MetricBase.REFERENCE_LEN_KEY_ARGUMENTS}
		gen_log_prob_key (str): The key of **log** probability over words.
			Default: ``gen_log_prob``.
		invalid_vocab (bool): Whether ``gen_log_prob`` contains :ref:`invalid vocab <vocab_ref>`.
			Default: ``False``.
		full_check (bool): Whether perform a full check on ``gen_log_prob`` to make sure the sum
			of probability is 1. Otherwise, a random check will be performed for efficiency.
			If pytorch is used, a full check is always performed and this argument will be ignored.
			Default: ``False``.

	Here is an example:

		>>> dl = cotk.dataloader.UbuntuCorpus('resources://Ubuntu_small')
		>>> reference_allvocabs_key="ref_allvocabs"
		>>> reference_len_key="ref_length"
		>>> gen_log_prob_key="gen_log_prob"
		>>> metric = cotk.metric.PerplexityMetric(dl,
		...	    reference_allvocabs_key="ref_allvocabs",
		...	    reference_len_key="ref_length",
		...	    gen_log_prob_key="gen_log_prob")
		>>> data = {
		...	    reference_allvocabs_key: [[2, 10, 64, 851, 3], [2, 10, 48, 851, 3]],
		...	    # reference_allvocabs_key: [["<go>", "I", "like", "python", "<eos>"], ["<go>", "I", "use", "python", "<eos>"]],
		...     reference_len_key: [5, 5],
		...     gen_log_prob_key: [[[-11.31, -11.31,  -0.69, ..., -11.31, -11.31, -11.31],...],...] # shape == (batch, length, vocab_size)
		... }
		>>> metric.forword(data)
		>>> metric.close()
		{'perplexity': 81458.00000000006,
 		 'perplexity hashvalue': '7f9b88b8f9996f5d49a512258f250fbc56adee714952b2c696c0b36cce36f648'}
	'''

	_name = 'PerplexityMetric'
	_version = 1

	@hooks.hook_metric
	def __init__(self, dataloader, \
					   reference_allvocabs_key="ref_allvocabs", \
					   reference_len_key="ref_length", \
					   gen_log_prob_key="gen_log_prob", \
					   invalid_vocab=False, \
					   full_check=False \
			  ):
		super().__init__(self._name, self._version)
		self.dataloader = dataloader
		self.reference_allvocabs_key = reference_allvocabs_key
		self.reference_len_key = reference_len_key
		self.gen_log_prob_key = gen_log_prob_key
		self.word_loss = 0
		self.length_sum = 0
		self.invalid_vocab = invalid_vocab
		self.full_check = full_check
		self.engine_version = "unknown" # can be 'default', 'pytorch' when first forward time

		self.resp = []
		#self.resp_length = []
		self.gen_valid_log_prob = []
		self.gen_unk_log_prob = []

	def forward(self, data):
		'''Processing a batch of data. Smoothing will be performed for :ref:`invalid vocabs <vocab_ref>`.
		:ref:`Unknowns vocabs <vocab_ref>` will be ignored.

		Arguments:
			data (dict): A dict at least contains the following keys:

				{MetricBase.FORWARD_REFERENCE_ALLVOCABS_ARGUMENTS_WITH_TORCH}
				{MetricBase.FORWARD_REFERENCE_LEN_ARGUMENTS}
				* **data[gen_log_prob_key]** (list or :class:`numpy.ndarray` or :class:`torch.Tensor`):
				  The **log softmax** probability of the sentence generations model outputs.
				  A 3-d jagged or padded array of float.
				  Contains end token (eg:``<eos>``), but without start token (eg: ``<go>``).
				  Size: ``[batch_size, ~gen_sentence_length, vocab_size]`` for ``invalid_vocab = False``, or
				  ``[batch_size, ~gen_sentence_length, all_vocab_size]`` for ``invalid_vocab = True``,
				  where "~" means different sizes in this dimension is allowed.
				  If :class:`torch.Tensor` is used, the following data should also be
				  :class:`torch.Tensor`.

				Here is an example for data:

					>>> # all_vocab_list = ["<pad>", "<unk>", "<go>", "<eos>", "I", "have",
					>>> #   "been", "to", "China"]
					>>> data = {
					...     reference_allvocabs_key: [[2,4,3], [2,5,6,3]],
					...	    reference_len_key: [3,4],
					...	    gen_log_prob_key: [[[-3.80666249, -3.11351531, -2.7080502 , -2.42036813, -2.19722458,
							    -2.01490302, -1.86075234, -1.72722095, -1.60943791],...],...]
					... }
		Warning:
			``data[gen_log_prob_key]`` must be processed after log_softmax. That means,
			``np.sum(np.exp(gen_log_prob), -1)`` equals ``np.ones((batch_size, gen_sentence_length))``
		'''
		super().forward(data)
		resp_allvocabs = data[self.reference_allvocabs_key]
		resp_length = data[self.reference_len_key]
		gen_log_prob = data[self.gen_log_prob_key]

		if not isinstance(resp_allvocabs, (torch.Tensor, np.ndarray, list)):
			raise TypeError("Unknown type for resp_allvocabs.")
		if not isinstance(gen_log_prob, (torch.Tensor, np.ndarray, list)):
			raise TypeError("Unknown type for gen_log_prob")
		if not isinstance(resp_length, (list, np.ndarray)):
			raise TypeError("Unknown type for resp_length")

		if self.engine_version == "unknown":
			if isinstance(gen_log_prob, torch.Tensor):
				self.engine_version = "pytorch"
			else:
				self.engine_version = "normal"

		if (self.engine_version == "pytorch") != isinstance(gen_log_prob, torch.Tensor):
			raise TypeError("If you want to use pytorch, `gen_log_prob` \
				should always be torch.Tensor. It can't mix with list or numpy.ndarray.")

		if self.engine_version == "pytorch":
			if not isinstance(resp_allvocabs, torch.Tensor):
				resp_allvocabs = gen_log_prob.new_tensor(resp_allvocabs).long()
			with torch.no_grad():
				self._pytorch_forward(resp_allvocabs, resp_length, gen_log_prob)
		else:
			self._normal_forward(resp_allvocabs, resp_length, gen_log_prob)

	def _normal_forward(self, resp_allvocabs, resp_length, gen_log_prob):
		if len(resp_allvocabs) != len(resp_length) or len(resp_allvocabs) != len(gen_log_prob):
			raise ValueError("Batch num of arguments is not matched.")

		# perform random check to assert the probability is valid
		checkid = random.randint(0, len(resp_length)-1)
		if resp_length[checkid] < 2:
			raise ValueError("resp_length must no less than 2, because <go> and <eos> are always included.")
		checkrow = random.randint(0, resp_length[checkid]-2)

		random_check_expsum = np.sum(np.exp(gen_log_prob[checkid][checkrow]))
		if not np.isclose(random_check_expsum, 1):
			raise ValueError("data[gen_log_prob_key] must be processed after log_softmax. \
				gen_log_prob[%d][%d] exp sum is equal to %f." % (checkid, checkrow, \
				random_check_expsum))

		relevant_data = []
		for i, resp_len in enumerate(resp_length):
			if resp_len < 2:
				raise ValueError("resp_length must no less than 2, because <go> and <eos> are always included.")

			resp_now = np.array(resp_allvocabs[i][1:resp_len])
			gen_now = np.array(gen_log_prob[i])
			relevant_data.append(resp_now.tolist())

			if len(resp_now.shape) != 1:
				raise ValueError("resp_allvocabs need to be 2 dimension")
			if len(gen_now.shape) != 2:
				raise ValueError("gen_log_prob need to be 3 dimension")

			# perform full check to assert the probability is valid
			if self.full_check:
				expsum = np.sum(np.exp(gen_now[:resp_len-1]), -1)
				if not np.allclose(expsum, [1] * (resp_len - 1)):
					raise ValueError("data[gen_log_prob_key] must be processed after log_softmax.")

			if not self.invalid_vocab:
				if gen_now.shape[1] != self.dataloader.vocab_size:
					raise ValueError("The third dimension gen_log_prob should be equals to vocab_size when \
						invalid_vocab = False, \
						but %d != %d" % (gen_now.shape[1], self.dataloader.vocab_size))
			else:
				if gen_now.shape[1] != self.dataloader.all_vocab_size:
					raise ValueError("The third dimension gen_log_prob should be equals to all_vocab_size \
						when invalid_vocab = True, \
						but %d != %d" % (gen_now.shape[1], self.dataloader.all_vocab_size))

			resp = resp_now
			self.resp.append(resp)
			#self.resp_length.append(resp_len)

			resp_known = resp.copy()
			if not self.invalid_vocab:
				resp_known[resp_known >= self.dataloader.vocab_size] = self.dataloader.unk_id

			self.gen_valid_log_prob.append(gen_now[list(range(resp_len-1)), resp_known])
			self.gen_unk_log_prob.append(gen_now[:resp_len-1, self.dataloader.unk_id])

		self._hash_relevant_data(relevant_data)

	def _pytorch_forward(self, resp_allvocabs, resp_length, gen_log_prob):
		if len(resp_allvocabs) != len(resp_length) or len(resp_allvocabs) != len(gen_log_prob):
			raise ValueError("Batch num of arguments is not matched.")
		if len(resp_allvocabs.shape) != 2:
			raise ValueError("resp_allvocabs need to be 2 dimension")
		if len(gen_log_prob.shape) != 3:
			raise ValueError("gen_log_prob need to be 3 dimension")

		relevant_data = []
		for i, resp_len in enumerate(resp_length):
			if resp_len < 2:
				raise ValueError("resp_length must no less than 2, because <go> and <eos> are always included.")

			resp_now = resp_allvocabs[i, 1:resp_len]
			gen_now = gen_log_prob[i, :resp_len - 1]
			relevant_data.append(resp_now.tolist())

			# perform full check to assert the probability is valid
			expsum = gen_now.exp().sum(-1)
			if not expsum.allclose(torch.ones_like(expsum)):
				raise ValueError("data[gen_log_prob_key] must be processed after log_softmax.")

			if not self.invalid_vocab:
				if gen_now.shape[1] != self.dataloader.vocab_size:
					raise ValueError("The third dimension gen_log_prob should be equals to vocab_size when \
						invalid_vocab = False, \
						but %d != %d" % (gen_now.shape[1], self.dataloader.vocab_size))
			else:
				if gen_now.shape[1] != self.dataloader.all_vocab_size:
					raise ValueError("The third dimension gen_log_prob should be equals to all_vocab_size \
						when invalid_vocab = True, \
						but %d != %d" % (gen_now.shape[1], self.dataloader.all_vocab_size))

			resp_known = resp_now.clone()
			if not self.invalid_vocab:
				resp_known[resp_known >= self.dataloader.vocab_size] = self.dataloader.unk_id

			unk_id = self.dataloader.unk_id
			vocab_size = self.dataloader.vocab_size
			invalid_vocab_size = self.dataloader.all_vocab_size - vocab_size

			# calc normal vocab
			normal_mask = ((resp_now != unk_id) & (resp_now < vocab_size)).float()
			word_loss = -(gen_now.gather(-1, resp_known.unsqueeze(1))[:, 0] * normal_mask).sum()
			length_sum = normal_mask.sum()
			# calc invalid vocab
			# smoothing from unk
			invalid_mask = (resp_now >= vocab_size).float()
			invalid_log_prob = (gen_now[:, unk_id] - \
						(torch.ones_like(gen_now[:, unk_id]) * invalid_vocab_size).log()) * invalid_mask

			if self.invalid_vocab:
				extra_invalid_log_prob = gen_now.gather(-1, resp_now.unsqueeze(1))[:, 0] * invalid_mask
				word_loss -= ((invalid_log_prob.exp() + extra_invalid_log_prob.exp()).log() \
						* invalid_mask).sum()
			else:
				word_loss -= invalid_log_prob.sum()

			length_sum += invalid_mask.sum()

			self.word_loss += word_loss.tolist()
			self.length_sum += length_sum.tolist()

		self._hash_relevant_data(relevant_data)

	@classmethod
	def _run_f(cls, ele):
		'''Auxiliary function for computing perplexity:

		Returns:

			* tuple: sum of log perplexity and sum of sentence length.
		'''
		valid_log_prob, unk_log_prob, resp_now, \
				invalid_vocab, vocab_size, all_vocab_size, unk_id = ele

		# calc normal vocab
		normal_idx = np.where(np.logical_and(resp_now != unk_id, \
								resp_now < vocab_size))
		word_loss = -np.sum(valid_log_prob[normal_idx])
		length_sum = np.array(normal_idx).shape[1]
		# calc invalid vocab
		# smoothing from unk
		invalid_idx = np.where(resp_now >= vocab_size)
		invalid_log_prob = unk_log_prob[invalid_idx] - np.log(all_vocab_size - vocab_size)
		if invalid_vocab:
			extra_invalid_log_prob = valid_log_prob[invalid_idx]
			word_loss -= np.sum(np.log( \
					np.exp(invalid_log_prob) + np.exp(extra_invalid_log_prob) \
				))
		else:
			word_loss -= np.sum(invalid_log_prob)
		length_sum += np.array(invalid_idx).shape[1]

		return word_loss, length_sum

	@hooks.hook_metric_close
	def close(self):
		r'''
		Returns:
			(dict): Return a dict which contains

			* **perplexity**: perplexity value.
			* **perplexity hashvalue**: hash value for perplexity metric, same hash value stands
			  for same evaluation settings.
		'''
		res = super().close()

		if self.engine_version == "pytorch":
			# pytorch is finished when forward
			if self.length_sum == 0:
				raise RuntimeError("The metric has not been forwarded data correctly.")
		else:
			if not self.gen_valid_log_prob:
				raise RuntimeError("The metric has not been forwarded data correctly.")

			loader = self.dataloader
			tasks = ((self.gen_valid_log_prob[i], self.gen_unk_log_prob[i], self.resp[i], \
							self.invalid_vocab, loader.vocab_size, loader.all_vocab_size, loader.unk_id) \
							for i, _ in enumerate(self.gen_valid_log_prob))

			# Multiprocessing seems can't boost the speed
			# if len(self.gen_valid_log_prob) > 100:
			# 	pool = Pool(multiprocessing.cpu_count())
			# 	for ans in tqdm.tqdm(pool.imap_unordered(self.run_f, tasks, chunksize=20), \
			# 		total=len(self.gen_valid_log_prob)):
			# 		self.word_loss += ans[0]
			# 		self.length_sum += ans[1]
			# 	pool.close()
			# 	pool.join()
			# else:
			for ans in map(self._run_f, tasks):
				self.word_loss += ans[0]
				self.length_sum += ans[1]

			self.resp = []
			self.gen_valid_log_prob = []
			self.gen_unk_log_prob = []

		res.update({"perplexity": np.exp(self.word_loss / self.length_sum), \
				"perplexity hashvalue": self._hashvalue()})
		return res

class MultiTurnPerplexityMetric(MetricBase):
	'''Metric for calculating multi-turn perplexity.

	Arguments:
		{MetricBase.DATALOADER_ARGUMENTS}
		{MetricBase.MULTI_TURN_REFERENCE_ALLVOCABS_KEY_ARGUMENTS}
		{MetricBase.MULTI_TURN_REFERENCE_LEN_KEY_ARGUMENTS}
		gen_log_prob_key (str): The key of **log** probability over words.
			Default: ``multi_turn_gen_log_prob``.
		invalid_vocab (bool): whether ``gen_log_prob`` contains :ref:`invalid vocab <vocab_ref>`.
			Default: ``False``.
		full_check (bool): whether perform full checks on ``gen_log_prob`` to make sure the sum
			of probability is 1. Otherwise, a random check will be performed for efficiency.
			Default: ``False``.

	Here is an example:

		>>> dl = cotk.dataloader.UbuntuCorpus('resources://Ubuntu_small')
		>>> multi_turn_reference_allvocabs_key = "multi_turn_ref_allvocabs"
		>>> multi_turn_reference_len_key = "multi_turn_ref_length"
		>>> multi_turn_gen_log_prob_key = "multi_turn_gen_log_prob"
		>>> metric = cotk.metric.MultiTurnPerplexityMetric(dl,
		...	    multi_turn_reference_allvocabs_key="multi_turn_ref_allvocabs",
		...	    multi_turn_reference_len_key="multi_turn_ref_length",
		...	    multi_turn_gen_log_prob_key="multi_turn_gen_log_prob")
		>>> data = {
		...	    multi_turn_reference_allvocabs_key: [[[2, 10, 64, 851, 3], [2, 10, 64, 479, 3]], [[2, 10, 64, 279, 1460, 3]]],
		...     # multi_turn_reference_allvocabs_key = [[["<go>", "I", "like", "python", "<eos>"],
		...     # 	["<go>", "I", "like", "java", "<eos>"]],
		...     # 	[["<go>", "I", "like", "machine", "learning", "<eos>"]]]
		...
		...	    multi_turn_reference_len_key: [[5, 5], [6]],
		...	    multi_turn_gen_log_prob_key: [[[[-11.30784283, -11.30784283,  -0.69312263, ..., -11.30784283, -11.30784283, -11.30784283], ...], ...], ...]
		... }
		>>> metric.forword(data)
		>>> metric.close()
		{'perplexity': 81458.00000000006,
 		 'perplexity hashvalue': '3a7647507f2e0d05a235c1d3a29515dc8885650884d625a5b76d305541dca685'}
	'''

	_name = 'MultiTurnPerplexityMetric'
	_version = 1

	@hooks.hook_metric
	def __init__(self, dataloader, multi_turn_reference_allvocabs_key="multi_turn_ref_allvocabs", \
					   multi_turn_reference_len_key="multi_turn_ref_length", \
					   multi_turn_gen_log_prob_key="multi_turn_gen_log_prob", \
					   invalid_vocab=False, \
					   full_check=False \
			  ):
		super().__init__(self._name, self._version)
		self.dataloader = dataloader
		self.multi_turn_reference_allvocabs_key = multi_turn_reference_allvocabs_key
		self.multi_turn_reference_len_key = multi_turn_reference_len_key
		self.multi_turn_gen_log_prob_key = multi_turn_gen_log_prob_key
		self.invalid_vocab = invalid_vocab
		self.sub_metric = PerplexityMetric(dataloader, \
				reference_allvocabs_key="ref_allvocabs", \
				reference_len_key="ref_length", \
				gen_log_prob_key="gen_log_prob", \
				invalid_vocab=invalid_vocab, \
				full_check=full_check)

	def forward(self, data):
		'''Processing a batch of data.

		Arguments:
			data (dict): A dict at least contains the following keys:

				{MetricBase.FORWARD_MULTI_TURN_REFERENCE_ALLVOCABS_ARGUMENTS_WITH_TORCH}
				{MetricBase.FORWARD_MULTI_TURN_REFERENCE_LEN_ARGUMENTS}
				* **data[multi_turn_gen_log_prob_key]** (list or :class:`numpy.ndarray` or \
					:class:`torch.Tensor`):
				  The **log softmax** probability of the sentence generations model outputs.
				  A 4-d jagged or padded array. **log softmax** probability.
				  Contains end token (eg:``<eos>``), but without start token (eg: ``<go>``).
				  Size: ``[batch_size, ~gen_sentence_length, vocab_size]`` for ``invalid_vocab = False``, or
				  ``[batch_size, ~gen_sentence_length, all_vocab_size]` for ``invalid_vocab = True``,
				  where "~" means different sizes in this dimension is allowed.
				  If :class:`torch.Tensor` is used, the following data should also be
				  :class:`torch.Tensor`.

				Here is an example for data:

					>>> # all_vocab_list = ["<pad>", "<unk>", "<go>", "<eos>", "I", "have",
					>>> #   "been", "to", "China"]
					>>> data = {
					...     multi_turn_reference_allvocabs_key: [[[2,4,3], [2,5,6,3]], [[2,7,6,8,3]]],
					...	    multi_turn_reference_len_key: [[3, 4], [5]],
					...	    multi_turn_gen_log_prob_key: [[[[-3.80666249, -3.11351531, -2.7080502,
						        -2.42036813, -2.19722458, -2.01490302, -1.86075234, -1.72722095,
						        -1.60943791], ...], ...], ...]
					... }
		Warning:
			``data[multi_turn_gen_log_prob_key]`` must be processed after log_softmax. That means,
			``np.sum(np.exp(multi_turn_gen_log_prob_key), -1)`` equals
			``np.ones((batch_size, ~gen_sentence_length))``
		'''
		super().forward(data)
		reference_allvocabs = data[self.multi_turn_reference_allvocabs_key]
		length = data[self.multi_turn_reference_len_key]
		gen_log_prob = data[self.multi_turn_gen_log_prob_key]

		if not isinstance(reference_allvocabs, (torch.Tensor, np.ndarray, list)):
			raise TypeError("Unknown type for reference_allvocabs.")
		if not isinstance(length, (np.ndarray, list)):
			raise TypeError("Unknown type for length")
		if not isinstance(gen_log_prob, (torch.Tensor, list, np.ndarray)):
			raise TypeError("Unknown type for gen_log_prob")

		if len(length) != len(reference_allvocabs) or len(length) != len(gen_log_prob):
			raise ValueError("Batch num is not matched.")

		for i, sent_length in enumerate(length):
			# Pass turn as batch for sub_metric, the result will be same.
			turn_length = sent_length.index(0) if 0 in sent_length else len(sent_length)
			if len(reference_allvocabs[i]) < turn_length or len(gen_log_prob[i]) < turn_length:
				raise ValueError("Turn num is not matched.")
			self.sub_metric.forward({"ref_allvocabs": reference_allvocabs[i][:turn_length], \
					"ref_length": sent_length[:turn_length], \
					"gen_log_prob": gen_log_prob[i][:turn_length]})

	@hooks.hook_metric_close
	def close(self):
		r'''
		Returns:
			(dict): Return a dict which contains

			* **perplexity**: perplexity value.
			* **perplexity hashvalue**: hash value for perplexity metric, same hash value stands
			  for same evaluation settings.
		'''
		res = super().close()
		res.update(self.sub_metric.close())
		return res
