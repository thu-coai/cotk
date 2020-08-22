import copy
import random

import numpy as np
import pytest

from cotk.metric import SingleTurnDialogRecorder, \
	MultiTurnDialogRecorder, LanguageGenerationRecorder

from metric_base import *

def setup_module():
	random.seed(0)
	np.random.seed(0)

#pytestmark = pytest.mark.skip("all tests still WIP")

single_turn_dialog_recorder_test_parameter = generate_testcase(\
	(zip(test_dataloader), "add"),
	(zip(test_argument), "add"),
	(zip(test_shape, test_type), "multi"),
	(zip(test_batch_len), "add"),
	(zip(test_gen_len), "multi"),
	(zip(test_ref_len), "multi")
)

class TestSingleTurnDialogRecorder():
	default_post_key = "post_allvocabs"
	default_ref_key = "resp_allvocabs"
	default_gen_key = "gen"
	default_keywords = [default_post_key, default_ref_key, default_gen_key]

	def get_sen_from_index(self, dataloader, data, post_key=default_post_key, \
			reference_key=default_ref_key, gen_key=default_gen_key):
		ans = { \
			'post': [], \
			'resp': [], \
			'gen': [], \
			}
		for sen in data[post_key]:
			ans['post'].append(dataloader.convert_ids_to_sentence(sen[1:]))
		for sen in data[reference_key]:
			ans['resp'].append(dataloader.convert_ids_to_sentence(sen[1:]))
		for sen in data[gen_key]:
			ans['gen'].append(dataloader.convert_ids_to_sentence(sen))

		return ans

	@pytest.mark.parametrize('data_loader, argument, shape, type, batch_len, gen_len, ref_len', single_turn_dialog_recorder_test_parameter)
	def test_close(self, data_loader, argument, shape, type, batch_len, gen_len, ref_len):
		# 'default' or 'custom'
		# 'pad' or 'jag'
		# 'list' or 'array'
		# 'equal' or 'unequal'
		dataloader = FakeDataLoader()
		post_key, reference_key, gen_key = self.default_keywords \
			if argument == 'default' else ('pk', 'rk', 'gk')
		data = dataloader.get_data(post_key=post_key, reference_key=reference_key, gen_key=gen_key, \
								   to_list=(type == 'list'), pad=(shape == 'pad'),
								   gen_len=gen_len, ref_len=ref_len)
		_data = copy.deepcopy(data)
		if data_loader == 'field':
			dataloader = dataloader.get_default_field()
		if argument == 'default':
			sr = SingleTurnDialogRecorder(dataloader)
		else:
			sr = SingleTurnDialogRecorder(dataloader, post_key, reference_key, gen_key)

		if batch_len == 'unequal':
			data[post_key] = data[post_key][1:]
			_data = copy.deepcopy(data)
			with pytest.raises(ValueError, match='Batch num is not matched.'):
				sr.forward(data)
		else:
			sr.forward(data)
			assert sr.close() == self.get_sen_from_index(dataloader, data, post_key, reference_key, \
																			gen_key)
		assert same_dict(data, _data)


multi_turn_dialog_test_parameter = generate_testcase(\
	(zip(test_dataloader), "add"),
	(zip(test_argument), "add"),
	(zip(test_shape, test_type), "multi"),
	(zip(test_batch_len), "add"),
	(zip(['empty', 'non-empty']), "multi"),
	(zip(['empty', 'non-empty']), "multi"),
	(zip(test_turn_len), "add")
)

class TestMultiTurnDialogRecorder:
	default_ref_key = 'multi_turn_ref_allvocabs'
	default_gen_key = "multi_turn_gen"
	default_turn_len_key= "turn_length"
	default_keywords = [default_ref_key, default_gen_key, default_turn_len_key]
	def check(self, ans, dataloader, data, \
			  resp_key=default_ref_key, gen_key=default_gen_key, turn_length=default_turn_len_key):
		_ans = {'reference': [], 'gen': []}

		for i, resp_turn in enumerate(data[resp_key]):
			resp_now = []
			for j, resp in enumerate(resp_turn):
				t = dataloader.trim_in_ids(resp[1:])
				if data[turn_length] is None:
					if len(t) == 0:
						break
				elif j >= data[turn_length][i]:
					break
				resp_now.append(t)
			_ans['reference'].append(resp_now)

		for i, gen_turn in enumerate(data[gen_key]):
			gen_now = []
			for j, gen in enumerate(gen_turn):
				t = dataloader.trim_in_ids(gen)
				if data[turn_length] is None:
					if len(t) == 0:
						break
				elif j >= data[turn_length][i]:
					break
				gen_now.append(t)
			_ans['gen'].append(gen_now)

		assert len(ans['reference']) == len(_ans['reference'])
		assert len(ans['gen']) == len(_ans['gen'])
		for i, turn in enumerate(ans['reference']):
			assert len(_ans['reference'][i]) == len(turn)
		for i, turn in enumerate(ans['gen']):
			assert len(_ans['gen'][i]) == len(turn)

	@pytest.mark.parametrize( \
		'data_loader, argument, shape, type, batch_len, gen_len, ref_len, turn_len', multi_turn_dialog_test_parameter)
	def test_close(self, data_loader, argument, shape, type, batch_len, gen_len, ref_len, turn_len):
		# 'default' or 'custom'
		# 'pad' or 'jag'
		# 'list' or 'array'
		# 'equal' or 'unequal'
		# 'random', 'non-empty', 'empty'
		# 'random', 'non-empty', 'empty'
		dataloader = FakeMultiDataloader()
		reference_key, gen_key, turn_len_key = self.default_keywords \
			if argument == 'default' else ('rk', 'gk', 'tk')
		data = dataloader.get_data(turn_len_key=turn_len_key, reference_key=reference_key, gen_key=gen_key, \
								   to_list=(type == 'list'), pad=(shape == 'pad'), \
								   gen_len=gen_len, ref_len=ref_len)
		_data = copy.deepcopy(data)
		if data_loader == 'field':
			dataloader = dataloader.get_default_field()
		if argument == 'default':
			mtbr = MultiTurnDialogRecorder(dataloader)
		else:
			mtbr = MultiTurnDialogRecorder(dataloader, reference_key, gen_key,
										   turn_len_key)

		if batch_len == 'unequal':
			data[reference_key] = data[reference_key][1:]
			_data = copy.deepcopy(data)
			with pytest.raises(ValueError, match='Batch num is not matched.'):
				mtbr.forward(data)
		else:
			if turn_len == 'unequal':
				data[reference_key][0] = data[reference_key][0][1:]
				with pytest.raises(ValueError, match=r"Reference turn num \d* != gen turn num \d*."):
					mtbr.forward(data)
				return
			else:
				mtbr.forward(data)
				self.check(mtbr.close(), dataloader, \
					data, reference_key, gen_key, turn_len_key)

		assert same_dict(data, _data)


language_generation_test_parameter = generate_testcase(\
	(zip(test_dataloader), "add"),
	(zip(test_argument), "add"),
	(zip(test_shape, test_type), "multi"),
	(zip(test_gen_len), "multi"),
)


class TestLanguageGenerationRecorder():
	def get_sen_from_index(self, dataloader, data, gen_key='gen'):
		ans = []
		for sen in data[gen_key]:
			ans.append(dataloader.convert_ids_to_sentence(sen))
		return ans

	@pytest.mark.parametrize('data_loader, argument, shape, type, gen_len', language_generation_test_parameter)
	def test_close(self, data_loader, argument, shape, type, gen_len):
		# 'default' or 'custom'
		# 'pad' or 'jag'
		# 'list' or 'array'
		# 'equal' or 'unequal'
		dataloader = FakeDataLoader()
		gen_key = 'gen' \
			if argument == 'default' else 'gk'
		data = dataloader.get_data(gen_key=gen_key, \
								   to_list=(type == 'list'), pad=(shape == 'pad'),
								   gen_len=gen_len)
		_data = copy.deepcopy(data)
		if data_loader == 'field':
			dataloader = dataloader.get_default_field()
		if argument == 'default':
			lg = LanguageGenerationRecorder(dataloader)
		else:
			lg = LanguageGenerationRecorder(dataloader, gen_key)

		lg.forward(data)
		assert lg.close()['gen'] == self.get_sen_from_index(dataloader, data, gen_key)
		assert same_dict(data, _data)
