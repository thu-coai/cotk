import numpy as np

from .wordvector import WordVector

class Glove(WordVector):
	def __init__(self, file_path):
		super().__init__()
		self.file_path = file_path

	def load(self, n_dims, vocab_list):
		raw_word2vec = {}
		if self.file_path:
			with open(self.file_path, "r") as f:
				lines = f.readlines()
			for l in lines:
				w, vec = l.split(" ", 1)
				raw_word2vec[w] = vec

		wordvec = []
		oov_cnt = 0
		for v in vocab_list:
			str_vec = raw_word2vec.get(v, None)
			vec = np.random.randn(n_dims) * 0.1
			if str_vec is None:
				oov_cnt += 1
			else:
				tmp = np.fromstring(str_vec, sep=" ")
				vec[:len(tmp)] = tmp
			wordvec.append(vec)
		print("wordvec cannot cover %f vocab" % (float(oov_cnt)/len(vocab_list)))
		return np.array(wordvec)
