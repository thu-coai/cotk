'''
A module for GloVe
'''
import os.path
import numpy as np

from .wordvector import WordVector
from .._utils.file_utils import get_resource_file_path

class Glove(WordVector):
	r'''GloVe is pre-trained word vector named `Global Vectors for Word Representation`.

	References:

		[1] Jeffrey Pennington, Richard Socher, and Christopher D. Manning. 2014.
		GloVe: Global Vectors for Word Representation.

	Arguments:
		file_id (str): a str indicates the source of GloVe word vectors. If it is local file,
			it can be a directory contains 'glove.txt' or just a text file.
			Default: ``resources://Glove300d``.	A 300d glove is downloaded and cached.
	'''
	def __init__(self, file_id="resources://Glove300d"):
		super().__init__()
		if file_id is not None:
			self.file_id = file_id
			self.file_path = get_resource_file_path(file_id)
		else:
			self.file_id = self.file_path = None

	def _load_raw_word2vec(self):
		raw_word2vec = {}
		if self.file_path:
			file_path = self.file_path
			if os.path.isdir(file_path):
				file_path = "%s/glove.txt" % (file_path)
			with open(file_path, 'r', encoding='utf-8') as glove_file:
				lines = glove_file.readlines()
			for line in lines:
				word, vec = line.split(" ", 1)
				raw_word2vec[word] = vec
		return raw_word2vec

	def load_matrix(self, n_dims, vocab_list, mean=0, std=0.1, default_embeddings=None):
		r'''
		Refer to :meth:`.WordVector.load`.
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

	def load_dict(self, vocab_list):
		r'''
		Refer to :meth:`.WordVector.load_pretrain_embed`.
		'''
		raw_word2vec = self._load_raw_word2vec()

		word2vec = {}
		for vocab in vocab_list:
			str_vec = raw_word2vec.get(vocab, None)
			if str_vec is not None:
				word2vec[vocab] = np.fromstring(str_vec, sep=" ")
		return word2vec
