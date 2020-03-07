'''
A module for word vector
'''
import numpy as np
from .._utils.metaclass import DocStringInheritor, LoadClassInterface
from typing import List, Dict, Union, Optional, Any


class WordVector(LoadClassInterface, metaclass=DocStringInheritor):
	r'''Base of all word vector loader
	'''
	def __init__(self):
		pass

	def load_matrix(self, n_dims: int, vocab_list: List[str], mean: float=0, std: float=0.1, default_embeddings: Any=None) -> np.ndarray:
		'''Load pretrained word vector and return a numpy 2-d array. The ith row is the feature
		of the ith word in ``vocab_list``. If some feature is not included in pretrained
		word vector, it will be initialized by:

		* ``default_embeddings``, if it is not ``None``.
		* normal distribution with ``mean`` and ``std``, otherwise.

		Arguments:
			n_dims (int): specify the dimension size of word vector. If ``n_dims``
				is bigger than size of pretrained word vector, the rest embedding will be
				initialized by ``default_embeddings`` or a normal distribution.
			vocab_list (list): specify the vocab list used in data loader. If there
				is any word not appeared in pretrained word vector, the embedding will be
				initialized by ``default_embeddings`` or a normal distribution.
			mean (float, optional): The mean of normal distribution. Default: 0.
			std (float, optional): The standard deviation of normal distribution. Default: 0.1.
			default_embeddings (Any, optional): The default embeddings, it size should be
				``[len(vocab_list), ndims]``. Default: None, which indicates initializing
				the embeddings from the normal distribution with ``mean`` and ``std``


		Returns:
		
			(:class:`numpy.ndarray`): A  2-d array. Size:``[len(vocab_list), n_dims]``.
		'''
		raise NotImplementedError("WordVector.load_matrix is a virtual function.")

	def load_dict(self, vocab_list: List[str]) -> Dict[str, np.ndarray]:
		'''Load word vector and return a dict that maps words to vectors.

		Arguments:
			vocab_list (list): specify the vocab list used in data loader. If there
				is any word not appeared in pretrained word vector, the feature will
				not be returned.

		Returns:

			(dict): maps a word (str) to its pretrained embedding (:class:`numpy.ndarray`)
				where its shape is [ndims].
		'''
		raise NotImplementedError("WordVector.load_dict is a virtual function.")
