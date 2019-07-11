import pytest
import numpy as np

from cotk.metric import NgramFwBwPerplexityMetric
from cotk.models.ngram_language_model import KneserNeyInterpolated

from metric_base import *


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

	@pytest.mark.parametrize('order', [1, 2, 3, 4])
	def test_perplexity(self, order):
		vocab = ['<go>', '<eos>', '<unk>', 'a', 'b', 'c', 'd']
		LM = KneserNeyInterpolated(order, vocab[0], vocab[1], vocab[2])
		corpus = []
		for i in range(20):
			corpus.append(list(np.random.choice(vocab[3:], np.random.randint(0, 10, 1)[0])))
		LM.fit(corpus[:10])

		assert LM.perplexity(corpus[10:]) > 0


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

		dataloader.data["test"][reference_key] = data[reference_key]
		fpm = NgramFwBwPerplexityMetric(dataloader, 4, reference_key)
		fpm.forward(data)
		res = fpm.close()

		data_shuffle = shuffle_instances(data, key_list)
		dataloader.data["test"][reference_key] = data_shuffle[reference_key]
		fpm_shuffle = NgramFwBwPerplexityMetric(dataloader, 4, reference_key)
		fpm_shuffle.forward(data_shuffle)
		res_shuffle = fpm_shuffle.close()

		assert res["fw-bw-ppl hashvalue"] == res_shuffle["fw-bw-ppl hashvalue"]

		for data_unequal in generate_unequal_data(data, key_list, dataloader.pad_id, \
												  reference_key, reference_is_3D=False):
			dataloader.data["test"][reference_key] = data_unequal[reference_key]
			fpm_unequal = NgramFwBwPerplexityMetric(dataloader, 4, reference_key)

			fpm_unequal.forward(data_unequal)
			res_unequal = fpm_unequal.close()
			assert res["fw-bw-ppl hashvalue"] != res_unequal["fw-bw-ppl hashvalue"]
		fpm_unequal = NgramFwBwPerplexityMetric(dataloader, 3, reference_key)
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
		dataloader.data["test"][reference_key] = data[reference_key]
		if argument == 'default':
			fpm = NgramFwBwPerplexityMetric(dataloader, 4, reference_key)
		else:
			fpm = NgramFwBwPerplexityMetric(dataloader, 4, reference_key, gen_key)

		fpm.forward(data)
		fpm.close()
