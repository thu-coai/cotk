import pytest
import numpy as np
import random
from unittest import mock
import tqdm

from cotk.metric import NgramFwBwPerplexityMetric
from cotk.models.ngram_language_model import KneserNeyInterpolated

from metric_base import *

pytestmark = pytest.mark.skip("all tests still WIP")

class TestNgramLM():
	@pytest.mark.parametrize('order', [1, 2, 3, 4])
	def test_score(self, order):
		vocab = ['<go>', '<eos>', '<unk>', 'a', 'b', 'c', 'd']
		valid_vocab = vocab[3:]
		LM = KneserNeyInterpolated(order, vocab[0], vocab[1], vocab[2])
		corpus = []
		for i in range(10):
			corpus.append(list(np.random.choice(valid_vocab, i)) + [valid_vocab[i % len(valid_vocab)]])
		LM.fit(corpus)

		context = tuple(np.random.choice(vocab, order + 1))
		with pytest.raises(RuntimeError, match="Provided context should be {}-gram.".format(order - 1)):
			LM.score(vocab[0], context)

		context = tuple(np.random.choice(vocab, order - 1))
		probs = []
		_vocab = vocab
		if order == 1:
			_vocab = vocab[2:]
		for word in _vocab:
			probs.append(LM.score(word, context))
			assert probs[-1] >= 0
		assert np.isclose(sum(probs), 1)

	@pytest.mark.parametrize('order, use_tqdm', zip([1, 1, 2, 2, 3, 3, 4, 4], [False, True] * 4))
	def test_perplexity(self, order, use_tqdm):
		# Test whether tqdm.tqdm is called, when sample >= 1000
		# and tqdm.tqdm is not called , when sample < 1000.
		# If tqdm.tqdm is called in `bcm.close`, `bleu_irl` will be replaced by random data.
		if use_tqdm:
			sample = random.randint(101, 200)
			assert sample > 100
		else:
			sample = random.randint(2, 100)
			assert 1 < sample <= 100
		fake_sent_lp = [random.random() for _ in range(sample)]
		vocab = ['<go>', '<eos>', '<unk>', 'a', 'b', 'c', 'd']
		LM = KneserNeyInterpolated(order, vocab[0], vocab[1], vocab[2])
		corpus = []
		for i in range(10 + sample):
			corpus.append(list(np.random.choice(vocab[3:], np.random.randint(0, 10, 1)[0])))
		LM.fit(corpus[:10])

		import multiprocessing
		with mock.patch('tqdm.tqdm', return_value=fake_sent_lp):
			with mock.patch('multiprocessing.pool.Pool'):
				if use_tqdm:
					assert np.isclose(LM.perplexity(corpus[10:]), np.exp(-sum(fake_sent_lp) / sum(map(len, corpus[10:]))))
					assert tqdm.tqdm.called
					if LM.cpu_count > 0:
						assert multiprocessing.pool.Pool.called
				else:
					assert LM.perplexity(corpus[10:]) > 0
					assert not tqdm.tqdm.called
					assert not multiprocessing.pool.Pool.called


fwbw_perplexity_test_parameter = generate_testcase(\
	(zip(test_argument), "add"),
	(zip(test_shape, test_type), "multi"),
	(zip(["non-empty"]), "multi"),
	(zip(["non-empty"]), "multi")
)
class TestNgramFwBwPerplexityMetric():
	@pytest.mark.parametrize('to_list, pad', [[True, False], [True, True], [False, True]])
	def test_hashvalue(self, to_list, pad):
		dataloader = FakeDataLoader()
		reference_key, gen_key = ('resp_allvocabs', 'gen')
		key_list = [reference_key, gen_key]
		data = dataloader.get_data(reference_key=reference_key, gen_key=gen_key, \
								   to_list=to_list, pad=pad, \
								   gen_len='non-empty', ref_len='non-empty')

		# dataloader.data["test"][reference_key] = data[reference_key]
		fpm = NgramFwBwPerplexityMetric(dataloader, 4, data[reference_key])
		fpm.forward(data)
		res = fpm.close()

		data_shuffle = shuffle_instances(data, key_list)
		# dataloader.data["test"][reference_key] = data_shuffle[reference_key]
		fpm_shuffle = NgramFwBwPerplexityMetric(dataloader, 4, data_shuffle[reference_key])
		fpm_shuffle.forward(data_shuffle)
		res_shuffle = fpm_shuffle.close()

		assert res["fw-bw-ppl hashvalue"] == res_shuffle["fw-bw-ppl hashvalue"]

		for data_unequal in generate_unequal_data(data, key_list, dataloader.pad_id, \
												  reference_key, reference_is_3D=False):
			# dataloader.data["test"][reference_key] = data_unequal[reference_key]
			fpm_unequal = NgramFwBwPerplexityMetric(dataloader, 4, data_unequal[reference_key])

			fpm_unequal.forward(data_unequal)
			res_unequal = fpm_unequal.close()
			assert res["fw-bw-ppl hashvalue"] != res_unequal["fw-bw-ppl hashvalue"]
		fpm_unequal = NgramFwBwPerplexityMetric(dataloader, 3, data[reference_key])
		fpm_unequal.forward(data)
		res_unequal = fpm_unequal.close()
		assert res["fw-bw-ppl hashvalue"] != res_unequal["fw-bw-ppl hashvalue"]

	@pytest.mark.parametrize('argument, shape, type, gen_len, ref_len', fwbw_perplexity_test_parameter)
	def test_close(self, argument, shape, type, gen_len, ref_len):
		# 'default' or 'custom'
		# 'pad' or 'jag'
		# 'list' or 'array'
		# 'equal' or 'unequal'
		# 'random', 'non-empty', 'empty'
		# 'random', 'non-empty', 'empty'
		dataloader = FakeDataLoader()
		reference_key, gen_key = ('resp_allvocabs', 'gen') \
			if argument == 'default' else ('rk', 'gk')
		data = dataloader.get_data(reference_key=reference_key, gen_key=gen_key, \
								   to_list=(type == 'list'), pad=(shape == 'pad'), \
								   gen_len=gen_len, ref_len=ref_len)
		# dataloader.data["test"][reference_key] = data[reference_key]
		if argument == 'default':
			fpm = NgramFwBwPerplexityMetric(dataloader, 4, data[reference_key])
		else:
			fpm = NgramFwBwPerplexityMetric(dataloader, 4, data[reference_key], gen_key)

		fpm.forward(data)
		fpm.close()

	def test_version(self):
		version_test(NgramFwBwPerplexityMetric, dataloader=FakeDataLoader())
