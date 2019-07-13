r"""
Containing some classes and functions about precision and recall evaluating results of models.
"""
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from .metric import _PrecisionRecallMetric

class BleuPrecisionRecallMetric(_PrecisionRecallMetric):
	r'''Metric for calculating sentence BLEU precision and recall.

	References:
		[1] Zhao, T., Zhao, R., & Eskenazi, M. (2017). Learning discourse-level diversity
		for neural dialog models using conditional variational autoencoders.
		arXiv preprint arXiv:1703.10960.

	Arguments:
		{_PrecisionRecallMetric.ARGUMENTS}
		ngram (int): Specifies using BLEU-ngram.
	'''

	def __init__(self, dataloader, \
				 ngram, \
				 generated_num_per_context, \
				 candidates_allvocabs_key='candidate_allvocabs', \
				 multiple_gen_key='multiple_gen'):
		super().__init__(dataloader, generated_num_per_context, candidates_allvocabs_key, \
				multiple_gen_key)
		if ngram not in range(1, 5):
			raise ValueError("ngram should belong to [1, 4]")
		self.ngram = ngram
		self.weights = [1 / ngram] * ngram
		self.res_prefix = 'BLEU-{}'.format(ngram)
		self._hash_relevant_data([ngram, generated_num_per_context])

	def _replace_unk(self, _input, _target=-1):
		'''Auxiliary function for replacing the unknown words:

		Arguments:
			_input (list): the references or hypothesis.
			_target: the target word index used to replace the unknown words.

		Returns:

			* list: processed result.
		'''
		output = []
		for ele in _input:
			output.append(_target if ele == self.dataloader.unk_id else ele)
		return output

	def _score(self, gen, reference):
		r'''Score function of BLEU-ngram precision and recall.

		Arguments:
			gen (list): list of generated word ids.
			reference (list): list of word ids of a reference.

		Returns:
			int: score \in [0, 1].

		Here is an Example:
			>>> gen = [4,5]
			>>> reference = [5,6]
			>>> self._score(gen, reference)
			0.150 # assume self.weights = [0.25,0.25,0.25,0.25]
		'''
		gen = self._replace_unk(gen)
		return sentence_bleu([reference], gen, self.weights, SmoothingFunction().method1)

class EmbSimilarityPrecisionRecallMetric(_PrecisionRecallMetric):
	r'''Metric for calculating cosine similarity precision and recall.

	References:
		[1] Zhao, T., Zhao, R., & Eskenazi, M. (2017). Learning discourse-level diversity
		for neural dialog models using conditional variational autoencoders.
		arXiv preprint arXiv:1703.10960.

	Arguments:
		{_PrecisionRecallMetric.ARGUMENTS}
		word2vec (dict): Maps a word (str) to its pretrained embedding (:class:`numpy.ndarray` or list)
		mode (str): Specifies the operation that computes the bag-of-word representation.
			Must be ``avg`` or ``extrema``:

			* ``avg`` : element-wise average word embeddings.
			* ``extrema`` : element-wise maximum word embeddings.

	'''

	def __init__(self, dataloader, \
				 word2vec, \
				 mode, \
				 generated_num_per_context, \
				 candidates_allvocabs_key='candidate_allvocabs', \
				 multiple_gen_key='multiple_gen'):
		super().__init__(dataloader, generated_num_per_context, \
			candidates_allvocabs_key, multiple_gen_key)
		if not isinstance(word2vec, dict):
			raise ValueError("word2vec has invalid type")
		if word2vec:
			embed_shape = np.array(list(word2vec.values())).shape
			if len(embed_shape) != 2 or embed_shape[1] == 0:
				raise ValueError("word embeddings have inconsistent embedding size or are empty")
		if mode not in ['avg', 'extrema']:
			raise ValueError("mode should be 'avg' or 'extrema'.")
		self.word2vec = word2vec
		self.mode = mode
		self.res_prefix = '{}-bow'.format(mode)
		self._hash_relevant_data([mode, generated_num_per_context] + \
				[(word, list(emb)) for word, emb in self.word2vec.items()])

	def _score(self, gen, reference):
		r'''Score function of cosine similarity precision and recall.

		Arguments:
			gen (list): list of generated word ids.
			reference (list): list of word ids of a reference.

		Returns:
			int: cosine similarity between two sentence embeddings \in [0, 1].

		Here is an Example:
			>>> gen = [4,5]
			>>> reference = [5,6]
			>>> self._score(gen, reference)
			0.135 # assume self.mode = 'avg'
		'''
		gen_vec = []
		ref_vec = []
		for word in self.dataloader.convert_ids_to_tokens(gen):
			if word in self.word2vec:
				gen_vec.append(self.word2vec[word])
		for word in self.dataloader.convert_ids_to_tokens(reference):
			if word in self.word2vec:
				ref_vec.append(self.word2vec[word])
		if not gen_vec or not ref_vec:
			return 0
		if self.mode == 'avg':
			gen_embed = np.average(gen_vec, 0)
			ref_embed = np.average(ref_vec, 0)
		else:
			gen_embed = np.max(gen_vec, 0)
			ref_embed = np.max(ref_vec, 0)
		cos = np.sum(gen_embed * ref_embed) / \
			  np.sqrt(np.sum(gen_embed * gen_embed) * np.sum(ref_embed * ref_embed))
		norm = (cos + 1) / 2
		return norm
