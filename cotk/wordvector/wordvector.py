'''
A module for word vector
'''
from .._utils.metaclass import DocStringInheritor, LoadClassInterface

class WordVector(LoadClassInterface, metaclass=DocStringInheritor):
	r'''Base of all word vector loader
	'''
	def __init__(self):
		pass

	def load_matrix(self, n_dims, vocab_list, mean=0, std=0.1, default_embeddings=None):
		'''Load pretrained word vector and return a numpy 2-d array. The ith row is the feature
		of the ith word in ``vocab_list``. If some feature is not included in pretrained
		word vector, it will be initialized by:

		* ``default_embeddings``, if it is not ``None``.
		* normal distribution with ``mean`` and ``std``, otherwise.

		Parameters:
			n_dims (int): specify the dimension size of word vector. If ``n_dims``
				is bigger than size of pretrained word vector, the rest embedding will be
				initialized by ``default_embeddings`` or a normal distribution.
			vocab_list (list): specify the vocab list used in data loader. If there
				is any word not appeared in pretrained word vector, the embedding will be
				initialized by ``default_embeddings`` or a normal distribution.
			mean: The mean of normal distribution. Default: 0.
			std: The standard deviation of normal distribution. Default: 0.1.
			default_embeddings: The default embeddings, it size should be
				``[len(vocab_list), ndims]``.


		Returns:

			(:class:`numpy.ndarray`): A  2-d array. Size:``[len(vocab_list), n_dims]``.
		'''
		raise NotImplementedError("WordVector.load_matrix is a virtual function.")

	def load_dict(self, vocab_list):
		'''Load word vector and return a dict that maps words to vectors.

		Parameters:
			vocab_list (list): specify the vocab list used in data loader. If there
				is any word not appeared in pretrained word vector, the feature will
				not be returned.

		Returns:

			(dict): maps a word (str) to its pretrained embedding (:class:`numpy.ndarray`)
				where its shape is [ndims].
		'''
		raise NotImplementedError("WordVector.load_dict is a virtual function.")
