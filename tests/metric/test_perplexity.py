import copy
import random
import numpy as np
import pytest
import torch

from cotk.metric import PerplexityMetric, MultiTurnPerplexityMetric

from metric_base import *

def setup_module():
	random.seed(0)
	np.random.seed(0)

pytestmark = pytest.mark.skip("all tests still WIP")

perplexity_test_parameter = generate_testcase(\
	(zip(test_argument), "add"),
	(zip(test_shape, test_type), "multi"),
	(zip(test_batch_len), "add"),
	(zip(test_check), "add"),
	(zip(test_ref_len), "multi"),
	(zip(test_ref_vocab), "multi"),
	(zip(test_gen_prob_vocab), "multi"),
	(zip(test_resp_len), "add"),
	(zip(test_include_invalid), "multi")
)


perplexity_test_engine_parameter = generate_testcase(\
	(zip(test_ref_vocab), "multi"),
	(zip(test_gen_prob_vocab), "multi"),
)

class TestPerplexityMetric():
	default_reference_key = 'ref_allvocabs'
	default_reference_len_key = 'ref_length'
	default_gen_prob_key = 'gen_log_prob'
	default_keywords = (default_reference_key, default_reference_len_key, default_gen_prob_key)

	def get_perplexity(self, input, dataloader, invalid_vocab=False, \
						reference_key=default_reference_key, \
						reference_len_key=default_reference_len_key, \
						gen_prob_key=default_gen_prob_key):
		length_sum = 0
		word_loss = 0
		for i in range(len(input[reference_key])):
			max_length = input[reference_len_key][i]

			for j in range(max_length - 1):
				vocab_now = input[reference_key][i][j + 1]
				if vocab_now == dataloader.unk_id:
					continue
				if vocab_now < dataloader.vocab_size:
					word_loss += -(input[gen_prob_key][i][j][vocab_now])
				else:
					invalid_log_prob = input[gen_prob_key][i][j][dataloader.unk_id] - \
									 np.log(dataloader.all_vocab_size - dataloader.vocab_size)
					if invalid_vocab:
						word_loss += -np.log(np.exp(invalid_log_prob) + \
											np.exp(input[gen_prob_key][i][j][vocab_now]))
					else:
						word_loss += -invalid_log_prob
				length_sum += 1
		# print('test_metric.word_loss: ', word_loss)
		# print('test_metric.length_sum: ',	 length_sum)
		return np.exp(word_loss / length_sum)

	@pytest.mark.parametrize('to_list, pad', [[True, False], [True, True], [False, True]])
	def test_hashvalue(self, to_list, pad):
		dataloader = FakeDataLoader()
		reference_key, reference_len_key, gen_prob_key = self.default_keywords
		key_list = [reference_key, reference_len_key, gen_prob_key]
		data = dataloader.get_data(reference_key=reference_key, \
								   reference_len_key=reference_len_key, gen_prob_key=gen_prob_key, \
								   to_list=to_list, pad=pad, \
								   gen_prob_check='no_check', ref_len='non-empty', \
								   ref_vocab='non-empty', gen_prob_vocab='all_vocab', \
								   resp_len='>=2')
		pm = PerplexityMetric(dataloader, invalid_vocab=True, full_check=False)
		pm_shuffle = PerplexityMetric(dataloader, invalid_vocab=True, full_check=False)

		data_shuffle = shuffle_instances(data, key_list)

		batches_shuffle = split_batch(data_shuffle, key_list, \
									  to_list=to_list, less_pad=pad, \
									  reference_key=reference_key, reference_is_3D=False)

		pm.forward(data)
		res = pm.close()

		for batch in batches_shuffle:
			pm_shuffle.forward(batch)
		res_shuffle = pm_shuffle.close()

		assert same_dict(res, res_shuffle, False)

		for data_unequal in generate_unequal_data(data, key_list, dataloader.pad_id, \
												  reference_key, reference_len_key, \
												  reference_is_3D=False):
			pm_unequal = PerplexityMetric(dataloader, invalid_vocab=True, full_check=False)

			pm_unequal.forward(data_unequal)
			res_unequal = pm_unequal.close()

			assert res['perplexity hashvalue'] != res_unequal['perplexity hashvalue']

	@pytest.mark.parametrize("ref_vocab, gen_prob_vocab", perplexity_test_engine_parameter)
	def test_same_result_with_pytorch_engine(self, ref_vocab, gen_prob_vocab):
		dataloader = FakeDataLoader()
		reference_key, reference_len_key, gen_prob_key = self.default_keywords
		data = dataloader.get_data(reference_key=reference_key, \
								   reference_len_key=reference_len_key, gen_prob_key=gen_prob_key, \
								   to_list=True, pad=True, \
								   gen_prob_check='no_check', ref_len='non-empty', \
								   ref_vocab=ref_vocab, gen_prob_vocab=gen_prob_vocab, \
								   resp_len='>=2')
		pm = PerplexityMetric(dataloader, invalid_vocab=gen_prob_vocab == "all_vocab", full_check=False)
		pm_shuffle = PerplexityMetric(dataloader, invalid_vocab=gen_prob_vocab == "all_vocab", full_check=False)
		pm_shuffle2 = PerplexityMetric(dataloader, invalid_vocab=gen_prob_vocab == "all_vocab", full_check=False)

		data_shuffle = copy.deepcopy(data)
		indices = list(range(len(data_shuffle[reference_key])))
		np.random.shuffle(indices)
		data_shuffle[reference_len_key] = list(np.array(data_shuffle[reference_len_key])[indices])
		data_shuffle[gen_prob_key] = torch.Tensor(np.array(data_shuffle[gen_prob_key])[indices])

		pm.forward(data)
		res = pm.close()

		data_shuffle[reference_key] = np.array(data_shuffle[reference_key])[indices]
		pm_shuffle.forward(data_shuffle)
		res_shuffle = pm_shuffle.close()

		data_shuffle[reference_key] = torch.LongTensor(data_shuffle[reference_key])
		pm_shuffle2.forward(data_shuffle)
		res_shuffle2 = pm_shuffle2.close()

		assert res['perplexity hashvalue'] == res_shuffle['perplexity hashvalue']
		assert res['perplexity hashvalue'] == res_shuffle2['perplexity hashvalue']
		assert np.isclose(res['perplexity'], res_shuffle['perplexity'])
		assert np.isclose(res['perplexity'], res_shuffle2['perplexity'])

	@pytest.mark.parametrize( \
		'argument, shape, type, batch_len, check, ref_len, ref_vocab, gen_prob_vocab, resp_len, include_invalid', \
		perplexity_test_parameter)
	def test_close(self, argument, shape, type, batch_len, check, \
				   ref_len, ref_vocab, gen_prob_vocab, resp_len, include_invalid):
		# 'default' or 'custom'
		# 'pad' or 'jag'
		# 'list' or 'array'
		# 'equal' or 'unequal'
		# 'random_check' or 'full_check' or 'no_check'
		# 'random', 'non-empty', 'empty'
		dataloader = FakeDataLoader()
		reference_key, reference_len_key, gen_prob_key = self.default_keywords \
			if argument == 'default' else ('ra', 'rl', 'glp')
		data = dataloader.get_data(reference_key=reference_key, \
								   reference_len_key=reference_len_key, gen_prob_key=gen_prob_key, \
								   to_list=(type == 'list'), pad=(shape == 'pad'), \
								   gen_prob_check=check, ref_len=ref_len, \
								   ref_vocab=ref_vocab, gen_prob_vocab=gen_prob_vocab, \
								   resp_len=resp_len)
		_data = copy.deepcopy(data)
		if argument == 'default':
			pm = PerplexityMetric(dataloader, invalid_vocab=include_invalid, full_check=(check=='full_check'))
		else:
			pm = PerplexityMetric(dataloader, reference_key, reference_len_key, gen_prob_key, \
								   invalid_vocab=include_invalid,  full_check=(check=='full_check'))

		if batch_len == 'unequal':
			data[reference_key] = data[reference_key][1:]
			_data = copy.deepcopy(data)
			with pytest.raises(ValueError, match='Batch num'):
				pm.forward(data)
		elif check == 'no_check':
			if resp_len == '<2':
				with pytest.raises(ValueError, match='resp_length must no less than 2,' \
													 ' because <go> and <eos> are always included.'):
					pm.forward(data)
			elif include_invalid != (gen_prob_vocab == 'all_vocab'):
				with pytest.raises(ValueError):
					pm.forward(data)
			else:
				pm.forward(data)
				assert np.isclose(pm.close()['perplexity'], \
								  self.get_perplexity(data, dataloader, include_invalid, \
													  reference_key, reference_len_key, gen_prob_key))
		else:
			with pytest.raises(ValueError, \
							   match=r'data\[gen_log_prob_key\] must be processed after log_softmax.'):
				pm.forward(data)
		assert same_dict(data, _data)

	def test_version(self):
		version_test(PerplexityMetric, dataloader=FakeDataLoader())

multiperplexity_test_parameter = generate_testcase(\
	(zip(test_argument), "add"),
	(zip(test_shape, test_type), "multi"),
	(zip(test_batch_len), "add"),
	(zip(test_check), "add"),
	(zip(test_ref_len), "multi"),
	(zip(test_ref_vocab), "multi"),
	(zip(test_gen_prob_vocab), "multi"),
	(zip(test_resp_len), "multi"),
	(zip(test_include_invalid), "multi")
)


class TestMultiTurnPerplexityMetric:
	default_reference_key = 'multi_turn_ref_allvocabs'
	default_reference_len_key = 'multi_turn_ref_length'
	default_gen_prob_key = 'multi_turn_gen_log_prob'
	default_keywords = (default_reference_key, default_reference_len_key, default_gen_prob_key)

	def get_perplexity(self, input, dataloader, invalid_vocab=False, \
					   reference_key=default_reference_key, \
					   reference_len_key=default_reference_len_key, \
					   gen_prob_key=default_gen_prob_key):
		length_sum = 0
		word_loss = 0
		for i in range(len(input[reference_key])):
			for turn in range(len(input[reference_key][i])):
				max_length = input[reference_len_key][i][turn]
				gen_prob_turn = input[gen_prob_key][i][turn]
				for j in range(max_length - 1):
					vocab_now = input[reference_key][i][turn][j + 1]
					if vocab_now == dataloader.unk_id:
						continue
					if vocab_now < dataloader.vocab_size:
						word_loss += -(gen_prob_turn[j][vocab_now])
					else:
						invalid_log_prob = gen_prob_turn[j][dataloader.unk_id] - \
										 np.log(dataloader.all_vocab_size - dataloader.vocab_size)
						if invalid_vocab:
							word_loss += -np.log(np.exp(invalid_log_prob) + \
												np.exp(gen_prob_turn[j][vocab_now]))
						else:
							word_loss += -invalid_log_prob
					length_sum += 1
		return np.exp(word_loss / length_sum)

	@pytest.mark.parametrize('to_list, pad', [[True, False], [True, True], [False, True]])
	def test_hashvalue(self, to_list, pad):
		dataloader = FakeMultiDataloader()
		reference_key, reference_len_key, gen_prob_key = self.default_keywords
		key_list = [reference_key, reference_len_key, gen_prob_key]
		data = dataloader.get_data(reference_key=reference_key, \
								   reference_len_key=reference_len_key, gen_prob_key=gen_prob_key, \
								   to_list=to_list, pad=pad, \
								   gen_prob_check='no_check', ref_len='non-empty', \
								   ref_vocab='non-empty', gen_prob_vocab='valid_vocab', \
								   resp_len=">=2")

		mtpm = MultiTurnPerplexityMetric(dataloader, invalid_vocab=False, full_check=False)
		mtpm_shuffle = MultiTurnPerplexityMetric(dataloader, invalid_vocab=False, full_check=False)

		data_shuffle = shuffle_instances(data, key_list)

		batches_shuffle = split_batch(data_shuffle, key_list, \
									  less_pad=pad, to_list=to_list, \
									  reference_key=reference_key, reference_is_3D=True)

		mtpm.forward(data)
		res = mtpm.close()

		for batch in batches_shuffle:
			mtpm_shuffle.forward(batch)
		res_shuffle = mtpm_shuffle.close()

		assert same_dict(res, res_shuffle, False)

		for data_unequal in generate_unequal_data(data, key_list, dataloader.pad_id, \
												  reference_key, reference_len_key, \
												  reference_is_3D=True):
			mtpm_unequal = MultiTurnPerplexityMetric(dataloader, invalid_vocab=False, full_check=False)

			mtpm_unequal.forward(data_unequal)
			res_unequal = mtpm_unequal.close()

			assert res['perplexity hashvalue'] != res_unequal['perplexity hashvalue']

	@pytest.mark.parametrize( \
		'argument, shape, type, batch_len, check, ref_len, ref_vocab, gen_prob_vocab, resp_len, include_invalid', \
		multiperplexity_test_parameter)
	def test_close(self, argument, shape, type, batch_len, check, \
				   ref_len, ref_vocab, gen_prob_vocab, resp_len, include_invalid):
		# 'default' or 'custom'
		# 'pad' or 'jag'
		# 'list' or 'array'
		# 'equal' or 'unequal'
		# 'random_check' or 'full_check' or 'no_check'
		# 'random', 'non-empty', 'empty'
		dataloader = FakeMultiDataloader()
		reference_key, reference_len_key, gen_prob_key = self.default_keywords \
			if argument == 'default' else ('rk', 'rl', 'gp')
		data = dataloader.get_data(reference_key=reference_key, \
								   reference_len_key=reference_len_key, gen_prob_key=gen_prob_key, \
								   to_list=(type == 'list'), pad=(shape == 'pad'), \
								   gen_prob_check=check, ref_len=ref_len, \
								   ref_vocab=ref_vocab, gen_prob_vocab=gen_prob_vocab, \
								   resp_len = resp_len)
		_data = copy.deepcopy(data)
		if argument == 'default':
			mtpm = MultiTurnPerplexityMetric(dataloader, \
											 invalid_vocab=include_invalid, full_check=(check=='full_check'))
		else:
			mtpm = MultiTurnPerplexityMetric(dataloader, reference_key, reference_len_key, gen_prob_key, \
								   invalid_vocab=include_invalid,  full_check=(check=='full_check'))

		if batch_len == 'unequal':
			data[reference_key] = data[reference_key][1:]
			_data = copy.deepcopy(data)
			with pytest.raises(ValueError, match='Batch num is not matched.'):
				mtpm.forward(data)
		elif check == 'no_check':
			if resp_len == '<2':
				with pytest.raises(ValueError, match='resp_length must no less than 2,' \
													 ' because <go> and <eos> are always included.'):
					mtpm.forward(data)
			elif include_invalid != (gen_prob_vocab == 'all_vocab'):
				with pytest.raises(ValueError):
					mtpm.forward(data)
			else:
				mtpm.forward(data)
				assert np.isclose(mtpm.close()['perplexity'], \
								  self.get_perplexity(data, dataloader, include_invalid, \
													  reference_key, reference_len_key, gen_prob_key))
		else:
			with pytest.raises(ValueError, \
							   match=r'data\[gen_log_prob_key\] must be processed after log_softmax.'):
				mtpm.forward(data)
		assert same_dict(data, _data)

	def test_version(self):
		version_test(MultiTurnPerplexityMetric, dataloader=FakeMultiDataloader())
