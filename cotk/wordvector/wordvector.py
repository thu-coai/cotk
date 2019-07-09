'''
A module for word vector
'''
from .._utils.metaclass import DocStringInheritor, LoadClassInterface

class WordVector(LoadClassInterface, metaclass=DocStringInheritor):
	r'''Base of all word vector loader
	'''
	def __init__(self):
		pass

	def load(self, n_dims, vocab_list):
		'''Load word vector and return a numpy array. The ith row is the feature
		of the ith word in ``vocab_list``.

		Parameters:
			n_dims (int): specify the dimension size of word vector. If ``n_dims``
				is bigger than pretrained word vector, the rest feature will be
				randomly initialized by normal distriution.
			vocab_list (list): specify the vocab list used in data loader. If there
				is any word not appeared in pretrained word vector, the feature will
				be randomly initialized by normal distribution

		Returns:

			(:class:`numpy.ndarray`): Size:``[len(vocab_list), n_dims]``.
		'''
		raise NotImplementedError("WordVector.load is a virtual function.")

	def load_pretrained_embed(self, n_dims, vocab_list):
		'''Load word vector and return a dict that maps words to vectors.

		Parameters:
			n_dims (int): specify the dimension size of word vector. If ``n_dims``
				is bigger than pretrained word vector, the rest feature will be
				randomly initialized by normal distriution.
			vocab_list (list): specify the vocab list used in data loader. If there
				is any word not appeared in pretrained word vector, the feature will
				be randomly initialized by normal distribution

		Returns:

			(dict): maps a word (str) to its pretrained embedding (:class:`numpy.ndarray` or list).
		'''
		raise NotImplementedError("WordVector.load_pretrain_embed is a virtual function.")
