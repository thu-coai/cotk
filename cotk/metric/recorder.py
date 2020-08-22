"""
Containing some recorders.
"""
from typing import List, Dict, Any, Union
import numpy as np
from .metric import MetricBase

class SingleTurnDialogRecorder(MetricBase):
	'''A metric-like class for recording generated sentences and references.

	Arguments:
		{MetricBase.DATALOADER_ARGUMENTS}
		post_allvocabs_key (str, optional): The key of dialog posts with :ref:`allvocabs <vocabulary_ref>`.
			Default: ``post_allvocabs``.
		resp_allvocabs_key (str, optional): The key of dialog responses with :ref:`allvocabs <vocabulary_ref>`.
			Default: ``resp_allvocabs``.
		{MetricBase.GEN_KEY_ARGUMENTS}


	Here is an example:

		>>> post_allvocabs_key = "post_allvocabs"
		>>> resp_allvocabs_key = "resp_allvocabs"
		>>> gen_key = "gen"
		>>> dl = cotk.dataloader.UbuntuCorpus('resources://Ubuntu_small')
		>>> metric = cotk.metric.SingleTurnDialogRecorder(dl,
		...     post_allvocabs_key=post_allvocabs_key,
		...     resp_allvocabs_key=resp_allvocabs_key,
		...     gen_key=gen_key)
		>>> data = {
		...     post_allvocabs_key: [[2, 4, 64, 739, 3, 0], [2, 4, 50, 739, 378, 3]],
		...     # post_allvocabs_key: [["<go>", "I", "like", "python", "<eos>", "<pad>"], ["<go>", "I", "use", "python", "most", "<eos>"]],
		...
		...	    resp_allvocabs_key: [[2, 4, 1193, 445, 3], [2, 739, 15, 2173, 3]],
		...	    # resp_allvocabs_key: [["<go>", "I", "prefer", "java", "<eos>"], ["<go>", "python", "is", "excellent", "<eos>"]],
		...
		...     gen_key: [[4, 64, 388], [739, 15, 3820, 3]],
		...     # gen_key: [["I", "like", "PHP"], ["python", "is", "powerful", "<eos>"]]
		... }
		>>> metric.forward(data)
		>>> metric.close()
		{'post': ['i like python', 'i use python most'],
 		 'resp': ['i prefer java', 'python is excellent'],
 		 'gen':  ['i like php', 'python is powerful']}
	'''

	_name = 'SingleTurnDialogRecorder'
	_version = 2
	def __init__(self, dataloader: Union["LanguageProcessing", "Sentence", "Session"], \
			post_allvocabs_key: str = "post_allvocabs", \
			resp_allvocabs_key: str = "resp_allvocabs", gen_key: str = "gen"):
		super().__init__(self._name, self._version)
		self.dataloader = dataloader
		self.post_allvocabs_key = post_allvocabs_key
		self.resp_allvocabs_key = resp_allvocabs_key
		self.gen_key = gen_key
		self.post_list = []
		self.resp_list = []
		self.gen_list = []

	def forward(self, data: Dict[str, Any]):
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
					...	    post_allvocabs_key: [[2,4,3,0], [2,5,6,3]],
					...	    resp_allvocabs_key: [[2,5,4,3], [2,6,3,0]],
					...	    gen_key: [[6,7,8], [4,5,3]]
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
			self.post_list.append(self.dataloader.convert_ids_to_sentence(post_sen[1:]))
			self.resp_list.append(self.dataloader.convert_ids_to_sentence(resp_allvocabs[i][1:]))
			self.gen_list.append(self.dataloader.convert_ids_to_sentence(gen[i]))

	def close(self) -> Dict[str, Any]:
		'''Return a dict which contains

			* **post**: a list of post sentences. A jagged 2-d array of int.
			  Size:``[batch_size, ~sent_length]``, where "~" means different
			  sizes in this dimension is allowed.
			* **resp**: a list of response sentences. A jagged 2-d array of int.
			  Size:``[batch_size, ~sent_length]``, where "~" means different
			  sizes in this dimension is allowed.
			* **gen**: A list of generated sentences. A jagged 2-d array of int.
			  Size:``[batch_size, ~sent_length]``, where "~" means different
			  sizes in this dimension is allowed.

			All sentences do not contain special tokens like ``<eos>``.
		'''
		res = super().close()
		res.update({"post": self.post_list, "resp": self.resp_list, "gen": self.gen_list})
		return res

class MultiTurnDialogRecorder(MetricBase):
	'''A metric-like class for recording generated sentences and references.

	Arguments:
		{MetricBase.MULTI_TURN_DATALOADER_ARGUMENTS}
		multi_turn_reference_allvocabs_key (str, optional): The key of dialog references with \
			:ref:`allvocabs <vocabulary_ref>`. Default: ``multi_turn_ref_allvocabs``.
		{MetricBase.MULTI_TURN_GEN_KEY_ARGUMENTS}
		{MetricBase.MULTI_TURN_LENGTH_KEY_ARGUMENTS}

	Here is an example:

		>>> multi_turn_reference_allvocabs_key = "multi_turn_ref_allvocabs"
		>>> multi_turn_gen_key = "multi_turn_gen"
		>>> turn_len_key = "turn_length"
		>>> dl = cotk.dataloader.UbuntuCorpus('resources://Ubuntu_small')
		>>> metric = cotk.metric.MultiTurnDialogRecorder(dl,
		...     multi_turn_reference_allvocabs_key=multi_turn_reference_allvocabs_key,
		...     multi_turn_gen_key=multi_turn_gen_key,
		...     turn_len_key=turn_len_key)
		>>> data = {
		...	    multi_turn_reference_allvocabs_key: [[[2, 4, 64, 739], [2, 4, 64, 445, 3]], [[4, 64, 283, 1436, 3]]],
		...     # multi_turn_reference_allvocabs_key = [[["<go>", "I", "like", "python"], ["<go>", "I", "like", "java", "<eos>"]],
		...     # 	[["I", "like", "machine", "learning", "<eos>"]]]
		...
		...	    turn_len_key: [2, 1],
		...     # turn_len_key: [len(multi_turn_reference_allvocabs_key[0]), len(multi_turn_reference_allvocabs_key[1])]
		...
		...	    multi_turn_gen_key: [[[739, 15, 2173, 3, 0, 0], [2, 388, 15, 387, 3, 0]], [[4, 64, 27937, 738, 2399, 3]]]
		...     # multi_turn_gen_key = [[["python", "is", "excellent", "<eos>", "<pad>, "<pad>"], ["<go>", "PHP", "is", "best", "<eos>", "<pad>"]],
		...     # 	[["I", "like", "natural", "language", "processing", "<eos>"]]]
		... }
		>>> metric.forward(data)
		>>> metric.close()
		{'reference': [['I like python', 'I like java'],
		 ['I like machine learning']],
		 'gen': [['python is excellent',
		 'PHP is best'],
		 ['I like natural language processing']]}

	'''
	_name = 'MultiTurnDialogRecorder'
	_version = 2
	def __init__(self, dataloader: Union["LanguageProcessing", "Session"],
			multi_turn_reference_allvocabs_key: str = "multi_turn_ref_allvocabs", \
			multi_turn_gen_key: str = "multi_turn_gen", \
			turn_len_key: str = "turn_length"):
		super().__init__(self._name, self._version)
		self.dataloader = dataloader
		self.multi_turn_reference_allvocabs_key = multi_turn_reference_allvocabs_key
		self.multi_turn_gen_key = multi_turn_gen_key
		self.turn_len_key = turn_len_key
		self.context_list = []
		self.reference_list = []
		self.gen_list = []

	def forward(self, data: Dict[str, Any]):
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
					...	    multi_turn_context_allvocabs_key: [[[2,4,3], [2,5,6,3]], [[2,7,6,8,3]]],
					...	    multi_turn_reference_allvocabs_key: [[[2,6,7,3], [2,5,3]], [[2,7,6,8,3]]],
					...	    multi_turn_gen_key: [[[6,7,8,3], [4,5,3]], [[7,3]]],
					...	    turn_len_key: [2,1]
					... }
		'''
		super().forward(data)
		reference_allvocabs = data[self.multi_turn_reference_allvocabs_key]
		gen = data[self.multi_turn_gen_key]
		turn_length = data[self.turn_len_key]

		if not isinstance(reference_allvocabs, (np.ndarray, list)):
			raise TypeError("Unknown type for reference_allvocabs")
		if not isinstance(gen, (np.ndarray, list)):
			raise TypeError("Unknown type for gen")
		if not isinstance(turn_length, (np.ndarray, list)):
			raise TypeError("Unknown type for turn_length")

		if len(turn_length) != len(reference_allvocabs) or \
			len(turn_length) != len(gen):
			raise ValueError("Batch num is not matched.")

		for i, _ in enumerate(reference_allvocabs):
			self.reference_list.append(self.dataloader.convert_multi_turn_ids_to_tokens( \
				reference_allvocabs[i], remove_special=True))
			self.gen_list.append(self.dataloader.convert_multi_turn_ids_to_tokens( \
				gen[i], remove_special=True))
			self.reference_list[-1] = [" ".join(toks) for toks in self.reference_list[-1]]
			self.gen_list[-1] = [" ".join(toks) for toks in self.gen_list[-1]]
			if len(self.reference_list[-1]) != len(self.gen_list[-1]):
				raise ValueError("Reference turn num %d != gen turn num %d." % \
						(len(self.reference_list[-1]), len(self.gen_list[-1])))

	def close(self) -> Dict[str, Any]:
		'''Return a dict which contains

			* **reference**: a list of response sentences. A jagged 3-d array of int.
			  Size:``[batch_size, ~turn_length, ~sent_length]``, where "~" means different
			  sizes in this dimension is allowed.
			* **gen**: a list of generated sentences. A jagged 3-d array of int.
			  Size:``[batch_size, ~turn_length, ~sent_length]``, where "~" means different
			  sizes in this dimension is allowed.

			All sentences do not contain special tokens like ``<eos>``.
		'''
		res = super().close()
		res.update({"reference": self.reference_list, "gen": self.gen_list})
		return res

class LanguageGenerationRecorder(MetricBase):
	'''A metric-like class for recorder generated sentences.

	Arguments:
		{MetricBase.DATALOADER_ARGUMENTS}
		{MetricBase.GEN_KEY_ARGUMENTS}

	Here is an example:

		>>> gen_key = "gen_key"
		>>> dl = cotk.dataloader.UbuntuCorpus('resources://Ubuntu_small')
		>>> metric = cotk.metric.LanguageGenerationRecorder(dl, gen_key=gen_key)
		>>> data = {
		...	    gen_key: [[2, 4, 64, 739, 3], [2, 4, 50, 739, 3]],
		...	    # gen_key: [["<go>", "I", "like", "python", "<eos>"], ["<go>", "I", "use", "python", "<eos>"]],
		... }
		>>> metric.forward(data)
		>>> metric.close()
		{'gen': ['I like python', 'I use python']}
	'''
	_name = 'LanguageGenerationRecorder'
	_version = 2
	def __init__(self, dataloader: Union["LanguageProcessing", "Sentence", "Session"], gen_key: str = "gen"):
		super().__init__(self._name, self._version)
		self.dataloader = dataloader
		self.gen_key = gen_key
		self.gen_list = []

	def forward(self, data: Dict[str, Any]):
		'''Processing a batch of data.

		Arguments:
			data (dict): A dict at least contains the following keys:

				{MetricBase.FORWARD_GEN_ARGUMENTS}

				Here is an example for data:

					>>> # all_vocab_list = ["<pad>", "<unk>", "<go>", "<eos>", "I", "have",
					>>> #   "been", "to", "China"]
					>>> data = {
					...	    gen_key: [[6,7,8,3], [4,5,3]]
					... }
		'''
		super().forward(data)
		gen = data[self.gen_key]

		if not isinstance(gen, (np.ndarray, list)):
			raise TypeError("Unknown type for gen")

		for sen in gen:
			self.gen_list.append(self.dataloader.convert_ids_to_sentence(sen))

	def close(self) -> Dict[str, Any]:
		'''Return a dict which contains

			* **gen**: a list of generated sentences. A jagged 2-d array of int.
			  Size:``[batch_size, ~sent_length]``, where "~" means different
			  sizes in this dimension is allowed.

			All sentences do not contain special tokens like ``<eos>``.
		'''
		res = super().close()
		res.update({"gen": self.gen_list})
		return res
