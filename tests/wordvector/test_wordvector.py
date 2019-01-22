import copy

import pytest
import numpy as np
from contk.dataloader import LanguageGeneration, MSCOCO
from contk.metric import MetricBase
from contk.wordvector.wordvector import WordVector
from contk.wordvector.gloves import Glove
import logging

class TestWordVector():
	def base_test_init(self, dl):
		assert isinstance(dl, WordVector)
		with pytest.raises(Exception):
			WordVector.load(None, None, None)
		WordVector.get_all_subclasses()
		assert WordVector.load_class('Glove') == Glove
		assert WordVector.load_class('not_subclass') == None

	def base_test_load(self, dl):
		vocab_list = ['first', 'second']
		n_dims = 3
		wordvec = dl.load(n_dims, vocab_list)
		assert isinstance(wordvec, np.ndarray)
		assert wordvec.shape == (len(vocab_list), n_dims)
		print(wordvec[1])
		assert wordvec[1].all() == np.array([0.1, 0.5, 0]).all()


		vocab_list = ['first', 'word_not_exist']
		n_dims = 3
		wordvec = dl.load(n_dims, vocab_list)
		assert isinstance(wordvec, np.ndarray)
		assert wordvec.shape == (len(vocab_list), n_dims)
		assert wordvec[0].all() == np.array([0.1, 1, 0.3]).all()

@pytest.fixture
def load_glove():
	def _load_glove():
		return Glove("./tests/wordvector/dummy_glove/glove.txt")
	return _load_glove

class TestGlove(TestWordVector):
	def test_init(self, load_glove):
		super().base_test_init(load_glove())

	def test_load(self, load_glove):
		super().base_test_load(load_glove())
