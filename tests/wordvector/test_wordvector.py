import copy

import pytest
import numpy as np
from cotk.dataloader import LanguageGeneration, MSCOCO
from cotk.metric import MetricBase
from cotk.wordvector.wordvector import WordVector
from cotk.wordvector.gloves import Glove
import logging

def setup_module():
	import random
	random.seed(0)
	import numpy as np
	np.random.seed(0)

class TestWordVector():
	def base_test_init(self, dl):
		assert isinstance(dl, WordVector)
		with pytest.raises(Exception):
			WordVector.load(None, None, None)
		WordVector.get_all_subclasses()
		assert WordVector.load_class('Glove') == Glove
		assert WordVector.load_class('not_subclass') == None

	def base_test_load(self, dl):
		vocab_list = ['the', 'of']
		n_dims = 300
		wordvec = dl.load(n_dims, vocab_list)
		assert isinstance(wordvec, np.ndarray)
		assert wordvec.shape == (len(vocab_list), n_dims)
		print(wordvec[1])
		assert wordvec[1][0] == -0.076947


		vocab_list = ['the', 'word_not_exist']
		n_dims = 300
		wordvec = dl.load(n_dims, vocab_list)
		assert isinstance(wordvec, np.ndarray)
		assert wordvec.shape == (len(vocab_list), n_dims)
		assert wordvec[0][0] == 0.04656

@pytest.fixture
def load_glove():
	def _load_glove():
		return Glove("./tests/wordvector/dummy_glove/300d")
	return _load_glove

class TestGlove(TestWordVector):
	def test_init(self, load_glove):
		super().base_test_init(load_glove())

	def test_load(self, load_glove):
		super().base_test_load(load_glove())
