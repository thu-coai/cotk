'''
A module for word vector
'''
import numpy as np
import os
from .._utils.metaclass import DocStringInheritor, LoadClassInterface
from typing import List, Dict, Union, Optional, Any

from ..file_utils import get_resource_file_path


class WordVector(LoadClassInterface, metaclass=DocStringInheritor):
	r'''Base of all word vector loader.
	'''

class GeneralWordVector(WordVector):
	r'''Bases: :class:`.dataloader.WordVector`

	This class is a general pretrained word vector.

	Arguments:
		{FILE_ID_DOCS} {_FILE_ID_DEFAULT}

	{INPUT_FORMAT}
	'''

	FILE_ID_DOCS = r'''
		file_id (str): A str indicates the source of word vectors. It can be local path (``"./data"``), a resource name
				(``"resources://dataset"``), or an url (``"http://test.com/dataset.zip"``).
				See :meth:`cotk.file_utils.get_resource_file_path` for further details.'''
	_FILE_ID_DEFAULT = ""

	INPUT_FORMAT = r'''
	Input Format
		A text file named ``wordvec.txt`` should be contained in the path. In the file, each word vec should be
		described in two lines. The first line is the word (or phrase), then the next line is multiple floats
		indicating the embedding.

		Example of ``wordvec.txt``:

		.. code-block:: none

			word
			0.0 1.0 -2.3
			phrases
			0.3 -1.2 3.4
	'''

	def __init__(self, file_id):
		super().__init__()
		if file_id is not None:
			self.file_id = file_id
			self.file_path = get_resource_file_path(file_id)
		else:
			self.file_id = self.file_path = None

	def _load_raw_word2vec(self) -> Dict[str, str]:
		'''Load raw word vectors from file.
		'''
		raw_word2vec = {}
		if self.file_path:
			file_path = self.file_path
			if os.path.isdir(file_path):
				file_path = "%s/wordvec.txt" % (file_path)
			with open(file_path, 'r', encoding='utf-8') as glove_file:
				lines = glove_file.readlines()
			for i in range(0, len(lines), 2):
				word = lines[i].strip()
				vec = lines[i+1].strip()
				raw_word2vec[word] = vec
		return raw_word2vec

	def load_matrix(self, n_dims: int, vocab_list: List[str], mean: float = 0.0, std: float = 0.1, \
			default_embeddings: Optional[Union[List, np.ndarray]] = None) -> np.ndarray:
		r'''Load pretrained word vector and return a numpy 2-d array. The ith row is the feature
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
		if default_embeddings is not None:
			if isinstance(default_embeddings, list):
				default_embeddings = np.array(default_embeddings)
			elif not isinstance(default_embeddings, np.ndarray):
				raise TypeError("Unkown type for default_embeddings")

			if default_embeddings.shape != (len(vocab_list), n_dims):
				raise ValueError("default_embeddings.shape should be equal to [len(vocab_list), n_dims]")

			default_embeddings = default_embeddings.copy()
		else:
			default_embeddings = np.random.randn(len(vocab_list), n_dims) * std + mean

		raw_word2vec = self._load_raw_word2vec()
		oov_cnt = 0
		have_warned = False
		for i, vocab in enumerate(vocab_list):
			str_vec = raw_word2vec.get(vocab, None)
			if str_vec is None:
				oov_cnt += 1
			else:
				tmp = np.fromstring(str_vec, sep=" ")
				if len(tmp) != n_dims and not have_warned:
					have_warned = True
					if len(tmp) > n_dims:
						print("Warning: Dimension of loaded wordvec is %d, but ``n_dims`` is set to %d. \
							The redundant dimension is trimmed." % (len(tmp), n_dims))
					else:
						print("Warning: Dimension of loaded wordvec is %d, but ``n_dims`` is set to %d. \
							The extra dimension is initialized by normal distribution (mean=0, std=0.1)."\
							% (len(tmp), n_dims))
				now_dims = min(len(tmp), n_dims)
				default_embeddings[i, :now_dims] = tmp[:now_dims]
		print("wordvec cannot cover %f vocab" % (float(oov_cnt)/len(vocab_list)))
		return default_embeddings

	def load_dict(self, vocab_list: List[str]) -> Dict[str, np.ndarray]:
		r'''Load word vector and return a dict that maps words to vectors.

		Arguments:
			vocab_list (list): specify the vocab list used in data loader. If there
				is any word not appeared in pretrained word vector, the feature will
				not be returned.

		Returns:

			(dict): maps a word (str) to its pretrained embedding (:class:`numpy.ndarray`)
				where its shape is [ndims].
		'''
		raw_word2vec = self._load_raw_word2vec()

		word2vec = {}
		for vocab in vocab_list:
			str_vec = raw_word2vec.get(vocab, None)
			if str_vec is not None:
				word2vec[vocab] = np.fromstring(str_vec, sep=" ")
		return word2vec


class Glove(GeneralWordVector):
	r'''Bases: :class:`.dataloader.GeneralWordVector`, :class:`.dataloader.WordVector`

	GloVe is pre-trained word vector named `Global Vectors for Word Representation`.

	References:

		[1] Jeffrey Pennington, Richard Socher, and Christopher D. Manning. 2014.
		GloVe: Global Vectors for Word Representation.

	Arguments:
		{FILE_ID_DOCS} {_FILE_ID_DEFAULT}

	'''
	_FILE_ID_DEFAULT = "Default: ``resources://Glove300d``.	A 300d glove is downloaded and cached."

	def __init__(self, file_id="resources://Glove300d"):
		super().__init__(file_id=file_id)
