import copy
import random
import operator
from unittest import mock

import tqdm
import numpy as np
import pytest

from cotk.metric import BleuCorpusMetric, SelfBleuCorpusMetric, \
	FwBwBleuCorpusMetric, MultiTurnBleuCorpusMetric

from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction

from metric_base import *

def setup_module():
	random.seed(0)
	np.random.seed(0)

pytestmark = pytest.mark.skip("all tests still WIP")

@pytest.mark.skip()
def test_bleu_bug():
	ref = [[[1, 3], [3], [4]]]
	gen = [[1]]
	with pytest.raises(ZeroDivisionError):
		corpus_bleu(ref, gen, smoothing_function=SmoothingFunction().method3)

bleu_test_parameter = generate_testcase(\
	(zip(test_argument), "add"),
	(zip(test_shape, test_type), "multi"),
	(zip(test_batch_len), "add"),
	(zip(test_gen_len), "multi"),
	(zip(test_ref_len), "multi")
)

class TestBleuCorpusMetric:
	default_reference_key = "ref_allvocabs"
	default_gen_key = "gen"
	default_keywords = [default_reference_key, default_gen_key]

	def get_bleu(self, dataloader, input, reference_key, gen_key):
		refs = []
		gens = []
		for gen_sen, resp_sen in zip(input[gen_key], input[reference_key]):
			gen_sen_processed = dataloader.trim(gen_sen)
			resp_sen_processed = dataloader.trim(resp_sen[1:])
			refs.append([resp_sen_processed])
			gens.append(gen_sen_processed)
		gens = replace_unk(gens)
		return corpus_bleu(refs, gens, smoothing_function=SmoothingFunction().method3)

	@pytest.mark.parametrize('to_list, pad', [[True, False], [True, True], [False, True]])
	def test_hashvalue(self, to_list, pad):
		dataloader = FakeDataLoader()
		reference_key, gen_key = self.default_keywords
		key_list = [reference_key, gen_key]
		data = dataloader.get_data(reference_key=reference_key, gen_key=gen_key, \
								   to_list=to_list, pad=pad, \
								   gen_len='non-empty', ref_len='non-empty')
		bcm = BleuCorpusMetric(dataloader)
		bcm_shuffle = BleuCorpusMetric(dataloader)

		data_shuffle = shuffle_instances(data, key_list)
		batches_shuffle = split_batch(data_shuffle, key_list, \
									  less_pad=pad, to_list=to_list, \
									  reference_key=reference_key, reference_is_3D=False)

		bcm.forward(data)
		res = bcm.close()

		for batch in batches_shuffle:
			bcm_shuffle.forward(batch)
		res_shuffle = bcm_shuffle.close()

		assert same_dict(res, res_shuffle, False)

		for data_unequal in generate_unequal_data(data, key_list, dataloader.pad_id, \
												  reference_key, reference_is_3D=False):
			bcm_unequal = BleuCorpusMetric(dataloader)

			bcm_unequal.forward(data_unequal)
			res_unequal = bcm_unequal.close()

			assert res['bleu hashvalue'] != res_unequal['bleu hashvalue']

	@pytest.mark.parametrize('argument, shape, type, batch_len, gen_len, ref_len', bleu_test_parameter)
	def test_close(self, argument, shape, type, batch_len, gen_len, ref_len):
		# 'default' or 'custom'
		# 'pad' or 'jag'
		# 'list' or 'array'
		# 'equal' or 'unequal'
		# 'random', 'non-empty', 'empty'
		# 'random', 'non-empty', 'empty'
		dataloader = FakeDataLoader()
		reference_key, gen_key = self.default_keywords \
			if argument == 'default' else ('rk', 'gk')
		data = dataloader.get_data(reference_key=reference_key, gen_key=gen_key, \
								   to_list=(type == 'list'), pad=(shape == 'pad'), \
								   gen_len=gen_len, ref_len=ref_len)
		_data = copy.deepcopy(data)
		if argument == 'default':
			bcm = BleuCorpusMetric(dataloader)
		else:
			bcm = BleuCorpusMetric(dataloader, reference_allvocabs_key=reference_key, gen_key=gen_key)

		if batch_len == 'unequal':
			data[reference_key] = data[reference_key][1:]
			_data = copy.deepcopy(data)
			with pytest.raises(ValueError, match='Batch num is not matched.'):
				bcm.forward(data)
		else:
			bcm.forward(data)
			assert np.isclose(bcm.close()['bleu'], self.get_bleu(dataloader, data, reference_key, gen_key))
		assert same_dict(data, _data)

	def test_version(self):
		version_test(BleuCorpusMetric, dataloader=FakeDataLoader())

	@pytest.mark.skip
	def test_bleu_bug(self):
		dataloader = FakeDataLoader()
		ref = [[2, 5, 3]]
		gen = [[5]]
		data = {self.default_reference_key: ref, self.default_gen_key: gen}
		bcm = BleuCorpusMetric(dataloader)

		with pytest.raises(ZeroDivisionError):
			bcm.forward(data)
			bcm.close()


self_bleu_test_parameter = generate_testcase( \
	(zip(test_argument), "add"),
	(zip(test_shape, test_type), "multi"),
	(zip(["non-empty"]), "multi"),
)


class TestSelfBleuCorpusMetric:
	def get_self_bleu(self, dataloader, input, gen_key):
		gens = []
		for gen_sen in input[gen_key]:
			gen_sen_processed = dataloader.trim(gen_sen)
			gens.append(gen_sen_processed)
		refs = copy.deepcopy(gens)
		_refs = replace_unk(refs)
		bleu_irl = []
		for i in range(len(gens)):
			bleu_irl.append(sentence_bleu(
				refs[:i] + refs[i + 1:], _refs[i], smoothing_function=SmoothingFunction().method1))
		return 1.0 * sum(bleu_irl) / len(bleu_irl)

	def test_hashvalue(self):
		dataloader = FakeDataLoader()
		gen_key = 'gen'
		key_list = [gen_key]
		data = dataloader.get_data(gen_key=gen_key, \
								   to_list=False, \
								   pad=True, \
								   gen_len='non-empty')
		bcm = SelfBleuCorpusMetric(dataloader)
		bcm_shuffle = SelfBleuCorpusMetric(dataloader)
		bcm_unequal = SelfBleuCorpusMetric(dataloader, sample=2)

		data_shuffle = shuffle_instances(data, key_list)
		batches_shuffle = split_batch(data_shuffle, key_list)

		bcm.forward(data)
		res = bcm.close()

		for batch in batches_shuffle:
			bcm_shuffle.forward(batch)
		res_shuffle = bcm_shuffle.close()

		assert same_dict(res, res_shuffle, exact_equal=False)

		bcm_unequal.forward(data)
		res_unequal = bcm_unequal.close()

		assert res['self-bleu hashvalue'] != res_unequal['self-bleu hashvalue']

	@pytest.mark.parametrize('argument, shape, type, gen_len, use_tqdm', generate_testcase( \
		(zip(test_argument), "add"),
		(zip(test_shape, test_type), "multi"),
		(zip(["non-empty"]), "multi"),
		(zip([True, False]), "multi")))
	def test_close(self, argument, shape, type, gen_len, use_tqdm):
		# 'default' or 'custom'
		# 'pad' or 'jag'
		# 'list' or 'array'
		# 'equal' or 'unequal'
		# 'random', 'non-empty', 'empty'
		# 'random', 'non-empty', 'empty'

		# Test whether tqdm.tqdm is called, when sample >= 1000
		# and tqdm.tqdm is not called , when sample < 1000.
		# If tqdm.tqdm is called in `bcm.close`, `bleu_irl` will be replaced by random data.
		if use_tqdm:
			sample = random.randint(1000, 2000)
			assert sample >= 1000
		else:
			sample = random.randint(2, 50)
			assert 1 < sample < 1000
		fake_bleu_irl = [random.random() for _ in range(sample)]

		import multiprocessing
		with mock.patch('tqdm.tqdm', return_value=fake_bleu_irl):
			with mock.patch('multiprocessing.pool.Pool'):
				dataloader = FakeDataLoader()
				gen_key = 'gen' \
					if argument == 'default' else 'gk'
				data = dataloader.get_data(gen_key=gen_key, \
										   to_list=(type == 'list'), \
										   pad=(shape == 'pad'), \
										   gen_len=gen_len,
										   batch=sample)
				_data = copy.deepcopy(data)
				if argument == 'default':
					bcm = SelfBleuCorpusMetric(dataloader, sample=4000)
				else:
					bcm = SelfBleuCorpusMetric(dataloader, gen_key, sample=4000)
				assert bcm.sample == 4000

				rng_state_st = random.getstate()
				bcm.forward(data)
				if use_tqdm:
					assert np.isclose(bcm.close()['self-bleu'], 1.0 * sum(fake_bleu_irl) / len(fake_bleu_irl))
					assert tqdm.tqdm.called
					if bcm.cpu_count > 1:
						assert multiprocessing.pool.Pool.called
				else:
					assert np.isclose(bcm.close()['self-bleu'], self.get_self_bleu(dataloader, _data, gen_key))
					assert not tqdm.tqdm.called
					assert not multiprocessing.pool.Pool.called
				assert bcm.sample == sample
				assert same_dict(data, _data)
				rng_state_ed = random.getstate()
				assert operator.eq(rng_state_st, rng_state_ed)

	def test_version(self):
		version_test(SelfBleuCorpusMetric, dataloader=FakeDataLoader())

# def test_self_bleu_bug(self):
#	 dataloader = FakeDataLoader()
#	 gen = [[1]]
#	 data = {'gen': gen}
#	 bcm = SelfBleuCorpusMetric(dataloader)

#	 with pytest.raises(ZeroDivisionError):
#		 bcm.forward(data)
#		 bcm.close()

fwbw_bleu_test_parameter = generate_testcase( \
	(zip(test_argument), "add"),
	(zip(test_shape, test_type), "multi"),
	(zip(["non-empty"]), "multi"),
	(zip(["non-empty"]), "multi"),
	(zip([True, False]), "add")
)


class TestFwBwBleuCorpusMetric:
	def get_bleu(self, dataloader, input, reference_key, gen_key):
		refs = []
		gens = []
		for gen_sen, resp_sen in zip(input[gen_key], input[reference_key]):
			gen_sen_processed = dataloader.trim(gen_sen)
			resp_sen_processed = dataloader.trim(resp_sen[1:])
			refs.append(resp_sen_processed)
			gens.append(gen_sen_processed)
		gens = replace_unk(gens)
		bleu_irl_bw, bleu_irl_fw = [], []
		for i in range(len(gens)):
			bleu_irl_fw.append(sentence_bleu(refs, gens[i], smoothing_function=SmoothingFunction().method1))
		for i in range(len(refs)):
			bleu_irl_bw.append(sentence_bleu(gens, refs[i], smoothing_function=SmoothingFunction().method1))

		fw_bleu = (1.0 * sum(bleu_irl_fw) / len(bleu_irl_fw))
		bw_bleu = (1.0 * sum(bleu_irl_bw) / len(bleu_irl_bw))
		return 2.0 * bw_bleu * fw_bleu / (fw_bleu + bw_bleu)

	@pytest.mark.parametrize('to_list, pad', [[True, False], [True, True], [False, True]])
	def test_hashvalue(self, to_list, pad):
		dataloader = FakeDataLoader()
		reference_key, gen_key = ('resp_allvocabs', 'gen')
		key_list = [reference_key, gen_key]
		data = dataloader.get_data(reference_key=reference_key, gen_key=gen_key, \
								   to_list=to_list, pad=pad, \
								   gen_len='non-empty', ref_len='non-empty')

		# dataloader.data["test"][reference_key] = data[reference_key]
		bcm = FwBwBleuCorpusMetric(dataloader, data[reference_key])
		bcm.forward(data)
		res = bcm.close()

		data_shuffle = shuffle_instances(data, key_list)
		# dataloader.data["test"][reference_key] = data_shuffle[reference_key]
		bcm_shuffle = FwBwBleuCorpusMetric(dataloader, data_shuffle[reference_key])
		bcm_shuffle.forward(data_shuffle)
		res_shuffle = bcm_shuffle.close()

		assert same_dict(res, res_shuffle, False)

		for data_unequal in generate_unequal_data(data, key_list, dataloader.pad_id, \
												  reference_key, reference_is_3D=False):
			# dataloader.data["test"][reference_key] = data_unequal[reference_key]
			bcm_unequal = FwBwBleuCorpusMetric(dataloader, data_unequal[reference_key])

			bcm_unequal.forward(data_unequal)
			res_unequal = bcm_unequal.close()
			assert res['fw-bw-bleu hashvalue'] != res_unequal['fw-bw-bleu hashvalue']
		bcm_unequal = FwBwBleuCorpusMetric(dataloader, data[reference_key], sample=2)
		bcm_unequal.forward(data)
		res_unequal = bcm_unequal.close()
		assert res['fw-bw-bleu hashvalue'] != res_unequal['fw-bw-bleu hashvalue']

	@pytest.mark.parametrize('argument, shape, type, gen_len, ref_len, use_tqdm', fwbw_bleu_test_parameter)
	def test_close(self, argument, shape, type, gen_len, ref_len, use_tqdm):
		# 'default' or 'custom'
		# 'pad' or 'jag'
		# 'list' or 'array'
		# 'equal' or 'unequal'
		# 'random', 'non-empty', 'empty'
		# 'random', 'non-empty', 'empty'
		if use_tqdm:
			sample = random.randint(1000, 2000)
			assert sample >= 1000
		else:
			sample = random.randint(2, 50)
			assert 1 < sample < 1000
		fake_bleu_irl_fw = [random.random() for _ in range(sample)]
		fake_bleu_irl_bw = [random.random() for _ in range(sample)]

		def get_fake_values(lists=(fake_bleu_irl_fw, fake_bleu_irl_bw), i=-1):
			def _get_fake_values(*_, **__):
				nonlocal i, lists
				i = (1 + i) % len(lists)
				return lists[i]

			return _get_fake_values

		import multiprocessing
		with mock.patch('tqdm.tqdm', side_effect=get_fake_values()):
			with mock.patch('multiprocessing.pool.Pool'):
				# tqdm.tqdm is replaced by _get_fake_values. It returns fake values.
				dataloader = FakeDataLoader()
				reference_key, gen_key = ('resp_allvocabs', 'gen') \
					if argument == 'default' else ('rk', 'gk')
				data = dataloader.get_data(reference_key=reference_key, gen_key=gen_key, \
										   to_list=(type == 'list'), pad=(shape == 'pad'), \
										   gen_len=gen_len, ref_len=ref_len, batch=sample)
				# dataloader.data["test"][reference_key] = data[reference_key]
				_data = copy.deepcopy(data)
				if argument == 'default':
					bcm = FwBwBleuCorpusMetric(dataloader, data[reference_key], sample=sample)
				else:
					bcm = FwBwBleuCorpusMetric(dataloader, data[reference_key], gen_key, sample=sample)

				rng_state_st = random.getstate()
				assert bcm.sample == sample
				bcm.forward(data)
				if use_tqdm:
					fw_bleu = (1.0 * sum(fake_bleu_irl_fw) / len(fake_bleu_irl_fw))
					bw_bleu = (1.0 * sum(fake_bleu_irl_bw) / len(fake_bleu_irl_bw))
					if fw_bleu + bw_bleu > 0:
						fw_bw_bleu = 2.0 * bw_bleu * fw_bleu / (fw_bleu + bw_bleu)
					else:
						fw_bw_bleu = 0
					assert np.isclose(bcm.close()['fw-bw-bleu'], fw_bw_bleu)
					assert tqdm.tqdm.called
					assert tqdm.tqdm.call_count == 2
					assert tqdm.tqdm() == fake_bleu_irl_fw
					assert tqdm.tqdm() == fake_bleu_irl_bw
					if bcm.cpu_count > 1:
						assert multiprocessing.pool.Pool.called

				else:
					assert np.isclose(bcm.close()['fw-bw-bleu'],
									  self.get_bleu(dataloader, _data, reference_key, gen_key))
					assert not tqdm.tqdm.called
					assert not multiprocessing.pool.Pool.called
				assert same_dict(data, _data)
				rng_state_ed = random.getstate()
				assert operator.eq(rng_state_st, rng_state_ed)

	def test_version(self):
		version_test(FwBwBleuCorpusMetric, dataloader=FakeDataLoader())

# def test_fwbwbleu_bug(self):
#	 dataloader = FakeDataLoader()
#	 ref = [[2, 1, 3]]
#	 gen = [[1]]
#	 reference_key = 'resp_allvocabs'
#	 data = {reference_key: ref, 'gen': gen}
#	 dataloader.data["test"][reference_key] = data[reference_key]
#	 bcm = FwBwBleuCorpusMetric(dataloader, reference_key)

#	 with pytest.raises(ZeroDivisionError):
#		 bcm.forward(data)
#		 bcm.close()


multi_bleu_test_parameter = generate_testcase( \
	(zip(test_argument), "add"),
	(zip(test_shape, test_type), "multi"),
	(zip(test_batch_len), "add"),
	(zip(test_gen_len), "multi"),
	(zip(test_ref_len), "multi")
)


class TestMultiTurnBleuCorpusMetric:
	default_reference_key = "reference_allvocabs"
	default_turn_len_key = "turn_length"
	default_gen_key = "multi_turn_gen"
	default_keywords = [default_reference_key, default_turn_len_key, default_gen_key]

	def get_bleu(self, dataloader, input, reference_key, gen_key):
		refs = []
		gens = []
		for i in range(len(input[reference_key])):
			for resp_sen, gen_sen in zip(input[reference_key][i], input[gen_key][i]):
				gen_sen_processed = dataloader.trim(gen_sen)
				resp_sen_processed = dataloader.trim(resp_sen)
				gens.append(gen_sen_processed)
				refs.append([resp_sen_processed[1:]])
		gens = replace_unk(gens)
		return corpus_bleu(refs, gens, smoothing_function=SmoothingFunction().method3)

	@pytest.mark.parametrize('to_list, pad', [[True, False], [True, True], [False, True]])
	def test_hashvalue(self, to_list, pad):
		dataloader = FakeMultiDataloader()
		reference_key, turn_len_key, gen_key = self.default_keywords
		key_list = [reference_key, turn_len_key, gen_key]
		data = dataloader.get_data(reference_key=reference_key, turn_len_key=turn_len_key, gen_key=gen_key, \
								   to_list=to_list, pad=pad, ref_len='non-empty', \
								   ref_vocab='non-empty')

		mtbcm = MultiTurnBleuCorpusMetric(dataloader)
		mtbcm_shuffle = MultiTurnBleuCorpusMetric(dataloader)

		data_shuffle = shuffle_instances(data, key_list)
		batches_shuffle = split_batch(data_shuffle, key_list, \
									  less_pad=pad, to_list=to_list, \
									  reference_key=reference_key, reference_is_3D=True)

		mtbcm.forward(data)
		res = mtbcm.close()

		for batch in batches_shuffle:
			mtbcm_shuffle.forward(batch)
		res_shuffle = mtbcm_shuffle.close()
		assert same_dict(res, res_shuffle, False)

		data_less_word = copy.deepcopy(data)
		for idx, turn_len in enumerate(data_less_word[turn_len_key]):
			if turn_len > 1:
				data_less_word[turn_len_key][idx] -= 1
		for data_unequal in [data_less_word] + generate_unequal_data(data, key_list, dataloader.pad_id, \
																	 reference_key=reference_key, reference_is_3D=True):
			mtbcm_unequal = MultiTurnBleuCorpusMetric(dataloader)

			mtbcm_unequal.forward(data_unequal)
			res_unequal = mtbcm_unequal.close()

			assert res['bleu hashvalue'] != res_unequal['bleu hashvalue']

	@pytest.mark.parametrize('argument, shape, type, batch_len, gen_len, ref_len', multi_bleu_test_parameter)
	def test_close(self, argument, shape, type, batch_len, gen_len, ref_len):
		# 'default' or 'custom'
		# 'pad' or 'jag'
		# 'list' or 'array'
		# 'equal' or 'unequal'
		# 'random', 'non-empty', 'empty'
		# 'random', 'non-empty', 'empty'
		dataloader = FakeMultiDataloader()
		reference_key, turn_len_key, gen_key = self.default_keywords \
			if argument == 'default' else ('rk', 'tlk', 'gk')
		data = dataloader.get_data(reference_key=reference_key, turn_len_key=turn_len_key, gen_key=gen_key, \
								   to_list=(type == 'list'), pad=(shape == 'pad'), \
								   gen_len=gen_len, ref_len=ref_len)
		_data = copy.deepcopy(data)
		if argument == 'default':
			mtbcm = MultiTurnBleuCorpusMetric(dataloader)
		else:
			mtbcm = MultiTurnBleuCorpusMetric(dataloader, multi_turn_reference_allvocabs_key=reference_key, \
											  multi_turn_gen_key=gen_key, turn_len_key=turn_len_key)

		if batch_len == 'unequal':
			data[reference_key] = data[reference_key][1:]
			_data = copy.deepcopy(data)
			with pytest.raises(ValueError, match='Batch num is not matched.'):
				mtbcm.forward(data)
		else:
			mtbcm.forward(data)
			assert np.isclose(mtbcm.close()['bleu'], self.get_bleu(dataloader, data, reference_key, gen_key))
		assert same_dict(data, _data)

	def test_version(self):
		version_test(MultiTurnBleuCorpusMetric, dataloader=FakeMultiDataloader())

	@pytest.mark.skip()
	def test_bleu(self):
		dataloader = FakeMultiDataloader()
		ref = [[[2, 5, 3]]]
		gen = [[[5]]]
		turn_len = [1]
		data = {self.default_reference_key: ref, self.default_gen_key: gen, self.default_turn_len_key: turn_len}
		mtbcm = MultiTurnBleuCorpusMetric(dataloader)

		with pytest.raises(ZeroDivisionError):
			mtbcm.forward(data)
			mtbcm.close()
