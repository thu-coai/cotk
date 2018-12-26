'''
A module for GloVe
'''
import numpy as np

from .wordvector import WordVector

class Glove(WordVector):
	r'''GloVe is pre-trained word vectors from
	Jeffrey Pennington, Richard Socher, and Christopher D. Manning. 2014.
	GloVe: Global Vectors for Word Representation.

	Arguments:
		file_path (str): a str indicates the dir of GloVe word vectors.
	'''
	def __init__(self, file_path):
		super().__init__()
		self.file_path = file_path

	def load(self, n_dims, vocab_list):
		r'''
		Refer to :meth:`.WordVector.load`.
		'''
		raw_word2vec = {}
		if self.file_path:
			with open(self.file_path, "r") as glove_file:
				lines = glove_file.readlines()
			for line in lines:
				word, vec = line.split(" ", 1)
				raw_word2vec[word] = vec

		wordvec = []
		oov_cnt = 0
		for vocab in vocab_list:
			str_vec = raw_word2vec.get(vocab, None)
			vec = np.random.randn(n_dims) * 0.1
			if str_vec is None:
				oov_cnt += 1
			else:
				tmp = np.fromstring(str_vec, sep=" ")
				now_dims = min(len(tmp), n_dims)
				vec[:now_dims] = tmp[:now_dims]
			wordvec.append(vec)
		print("wordvec cannot cover %f vocab" % (float(oov_cnt)/len(vocab_list)))
		return np.array(wordvec)
