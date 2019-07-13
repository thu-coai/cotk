r"""
Containing some recorders.
"""
import numpy as np
from .metric import MetricBase

class SingleTurnDialogRecorder(MetricBase):
	'''A metric-like class for recording generated sentences and references.

	Arguments:
		{MetricBase.DATALOADER_ARGUMENTS}
		post_allvocabs_key (str): The key of dialog posts with :ref:`allvocabs <vocab_ref>`.
			Default: ``post_allvocabs``.
		resp_allvocabs_key (str): The key of dialog responses with :ref:`allvocabs <vocab_ref>`.
			Default: ``resp_allvocabs``.
		{MetricBase.GEN_KEY_ARGUMENTS}
	'''
	def __init__(self, dataloader, post_allvocabs_key="post_allvocabs", \
			resp_allvocabs_key="resp_allvocabs", gen_key="gen"):
		super().__init__()
		self.dataloader = dataloader
		self.post_allvocabs_key = post_allvocabs_key
		self.resp_allvocabs_key = resp_allvocabs_key
		self.gen_key = gen_key
		self.post_list = []
		self.resp_list = []
		self.gen_list = []

	def forward(self, data):
		'''Processing a batch of data.

		Arguments:
			data (dict): A dict at least contains the following keys:

				{MetricBase.FORWARD_POST_ALLVOCABS_ARGUMENTS}
				{MetricBase.FORWARD_RESP_ALLVOCABS_ARGUMENTS}
				{MetricBase.FORWARD_GEN_ARGUMENTS}

				Here is an example for data:
					>>> # all_vocab_list = ["<pad>", "<unk>", "<go>", "<eos>", "I", "have",
					>>> #   "been", "to", "China"]
					>>> data = {
					...		post_allvocabs_key: [[2,4,3], [2,5,6,3]]
					...		resp_allvocabs_key: [[2,5,4,3], [2,6,3]]
					...		gen_key: [[6,7,8,3], [4,5,3]]
					... }
		'''
		super().forward(data)
		post_allvocabs = data[self.post_allvocabs_key]
		resp_allvocabs = data[self.resp_allvocabs_key]
		gen = data[self.gen_key]

		if not isinstance(post_allvocabs, (np.ndarray, list)):
			raise TypeError("Unknown type for post_allvocabs.")
		if not isinstance(resp_allvocabs, (np.ndarray, list)):
			raise TypeError("Unknown type for resp_allvocabs")
		if not isinstance(gen, (np.ndarray, list)):
			raise TypeError("Unknown type for gen")

		if len(post_allvocabs) != len(resp_allvocabs) or len(resp_allvocabs) != len(gen):
			raise ValueError("Batch num is not matched.")
		for i, post_sen in enumerate(post_allvocabs):
			self.post_list.append(self.dataloader.convert_ids_to_tokens(post_sen[1:]))
			self.resp_list.append(self.dataloader.convert_ids_to_tokens(resp_allvocabs[i][1:]))
			self.gen_list.append(self.dataloader.convert_ids_to_tokens(gen[i]))

	def close(self):
		'''
		Returns:
			(dict): Return a dict which contains

			* **post**: a list of post sentences. A jagged 2-d array of int.
			  Size:``[batch_size, ~sent_length]``, where "~" means different
			  sizes in this dimension is allowed.
			* **resp**: a list of response sentences. A jagged 2-d array of int.
			  Size:``[batch_size, ~sent_length]``, where "~" means different
			  sizes in this dimension is allowed.
			* **gen**: A list of generated sentences. A jagged 2-d array of int.
			  Size:``[batch_size, ~sent_length]``, where "~" means different
			  sizes in this dimension is allowed.
		'''
		res = super().close()
		res.update({"post": self.post_list, "resp": self.resp_list, "gen": self.gen_list})
		return res

class MultiTurnDialogRecorder(MetricBase):
	'''A metric-like class for recording generated sentences and references.

	Arguments:
		{MetricBase.DATALOADER_ARGUMENTS}
		multi_turn_context_allvocabs_key (str): The key of dialog context with
			:ref:`allvocabs <vocab_ref>`. Default: ``multi_turn_context_allvocabs``.
		multi_turn_reference_allvocabs_key (str): The key of dialog references with
			:ref:`allvocabs <vocab_ref>`. Default: ``multi_turn_ref_allvocabs``.
		{MetricBase.MULTI_TURN_GEN_KEY_ARGUMENTS}
		{MetricBase.MULTI_TURN_LENGTH_KEY_ARGUMENTS}
	'''
	def __init__(self, dataloader, multi_turn_context_allvocabs_key="multi_turn_context_allvocabs", \
			multi_turn_reference_allvocabs_key="multi_turn_ref_allvocabs", \
			multi_turn_gen_key="multi_turn_gen", \
			turn_len_key="turn_length"):
		super().__init__()
		self.dataloader = dataloader
		self.multi_turn_context_allvocabs_key = multi_turn_context_allvocabs_key
		self.multi_turn_reference_allvocabs_key = multi_turn_reference_allvocabs_key
		self.multi_turn_gen_key = multi_turn_gen_key
		self.turn_len_key = turn_len_key
		self.context_list = []
		self.reference_list = []
		self.gen_list = []

	def forward(self, data):
		'''Processing a batch of data.

		Arguments:
			data (dict): A dict at least contains the following keys:

				{MetricBase.FORWARD_MULTI_TURN_CONTEXT_ALLVOCABS_ARGUMENTS}
				{MetricBase.FORWARD_MULTI_TURN_REFERENCE_ALLVOCABS_ARGUMENTS}
				{MetricBase.FORWARD_MULTI_TURN_GEN_ARGUMENTS}
				{MetricBase.FORWARD_MULTI_TURN_LENGTH_ARGUMENTS}

				Here is an example for data:
					>>> # all_vocab_list = ["<pad>", "<unk>", "<go>", "<eos>", "I", "have",
					>>> #   "been", "to", "China"]
					>>> data = {
					...		multi_turn_context_allvocabs_key: [[[2,4,3], [2,5,6,3]], [[2,7,6,8,3]]]
					...		multi_turn_reference_allvocabs_key: [[[2,6,7,3], [2,5,3]], [[2,7,6,8,3]]]
					...		multi_turn_gen_key: [[[6,7,8,3], [4,5,3]], [[7,3]]]
					...		turn_len_key: [2,1]
					... }
		'''
		super().forward(data)
		context_allvocabs = data[self.multi_turn_context_allvocabs_key]
		reference_allvocabs = data[self.multi_turn_reference_allvocabs_key]
		gen = data[self.multi_turn_gen_key]
		turn_length = data[self.turn_len_key]

		if not isinstance(context_allvocabs, (np.ndarray, list)):
			raise TypeError("Unknown type for context_allvocabs.")
		if not isinstance(reference_allvocabs, (np.ndarray, list)):
			raise TypeError("Unknown type for reference_allvocabs")
		if not isinstance(gen, (np.ndarray, list)):
			raise TypeError("Unknown type for gen")
		if not isinstance(turn_length, (np.ndarray, list)):
			raise TypeError("Unknown type for turn_length")

		if len(turn_length) != len(context_allvocabs) or \
			len(turn_length) != len(reference_allvocabs) or \
			len(turn_length) != len(gen):
			raise ValueError("Batch num is not matched.")

		for i, context_sen in enumerate(context_allvocabs):
			self.context_list.append(self.dataloader.convert_multi_turn_ids_to_tokens( \
				np.array(context_sen), ignore_first_token=True))
			self.reference_list.append(self.dataloader.convert_multi_turn_ids_to_tokens( \
				np.array(reference_allvocabs[i]), turn_length=turn_length[i], ignore_first_token=True))
			self.gen_list.append(self.dataloader.convert_multi_turn_ids_to_tokens( \
				np.array(gen[i]), turn_length=turn_length[i]))

			if len(self.reference_list[-1]) != len(self.gen_list[-1]):
				raise ValueError("Reference turn num %d != gen turn num %d." % \
						(len(self.reference_list[-1]), len(self.gen_list[-1])))

	def close(self):
		'''
		Returns:
			(dict): Return a dict which contains

			* **context**: a list of post sentences. A jagged 3-d array of int.
			  Size:``[batch_size, ~turn_length, ~sent_length]``, where "~" means different
			  sizes in this dimension is allowed.
			* **reference**: a list of response sentences. A jagged 3-d array of int.
			  Size:``[batch_size, ~turn_length, ~sent_length]``, where "~" means different
			  sizes in this dimension is allowed.
			* **gen**: a list of generated sentences. A jagged 3-d array of int.
			  Size:``[batch_size, ~turn_length, ~sent_length]``, where "~" means different
			  sizes in this dimension is allowed.
		'''
		res = super().close()
		res.update({"context": self.context_list, "reference": self.reference_list, "gen": self.gen_list})
		return res

class LanguageGenerationRecorder(MetricBase):
	'''A metric-like class for recorder generated sentences.

	Arguments:
		{MetricBase.DATALOADER_ARGUMENTS}
		{MetricBase.GEN_KEY_ARGUMENTS}
	'''
	def __init__(self, dataloader, gen_key="gen"):
		super().__init__()
		self.dataloader = dataloader
		self.gen_key = gen_key
		self.gen_list = []

	def forward(self, data):
		'''Processing a batch of data.

		Arguments:
			data (dict): A dict at least contains the following keys:

				{MetricBase.FORWARD_GEN_ARGUMENTS}

				Here is an example for data:
					>>> # all_vocab_list = ["<pad>", "<unk>", "<go>", "<eos>", "I", "have",
					>>> #   "been", "to", "China"]
					>>> data = {
					...		gen_key: [[6,7,8,3], [4,5,3]]
					... }
		'''
		super().forward(data)
		gen = data[self.gen_key]

		if not isinstance(gen, (np.ndarray, list)):
			raise TypeError("Unknown type for gen")

		for sen in gen:
			self.gen_list.append(self.dataloader.convert_ids_to_tokens(sen))

	def close(self):
		'''
		Returns:
			(dict): Return a dict which contains

			* **gen**: a list of generated sentences. A jagged 2-d array of int.
			  Size:``[batch_size, ~sent_length]``, where "~" means different
			  sizes in this dimension is allowed.
		'''
		res = super().close()
		res.update({"gen": self.gen_list})
		return res
