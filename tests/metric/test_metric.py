import copy

import numpy as np
import pytest

from random import random, randrange
from contk.metric import MetricBase, PerlplexityMetric, MultiTurnPerplexityMetric, BleuCorpusMetric, \
	MultiTurnBleuCorpusMetric, SingleTurnDialogRecorder, MultiTurnDialogRecorder, LanguageGenerationRecorder, \
	MetricChain
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction


def test_bleu_bug():
	ref = [[[1, 3], [3], [4]]]
	gen = [[1]]
	with pytest.raises(ZeroDivisionError):
		corpus_bleu(ref, gen, smoothing_function=SmoothingFunction().method7)


class FakeDataLoader:
	def __init__(self):
		self.vocab_list = ['<pad>', '<unk>', '<go>', '<eos>', 'what', 'how', 'here', 'do']
		self.vocab_to_index = {'<pad>': 0, '<unk>': 1, '<go>': 2, '<eos>': 3, 'what': 4, 'how': 5, 'here': 6, 'do': 7}
		self.vocab_size = 8

	def trim_before_target(self, lists, target):
		try:
			lists = lists[:lists.index(target)]
		except ValueError:
			pass
		return lists

	def trim_index(self, index):
		print(index)
		index = self.trim_before_target(list(index), 3)
		idx = len(index)
		while idx > 0 and index[idx - 1] == 0:
			idx -= 1
		index = index[:idx]
		return index

	def multi_turn_sen_to_index(self, session):
		return list(map(lambda sent: list(map( \
			lambda word: self.word2id.get(word, self.unk_id), sent)), \
						session))

	def multi_turn_trim_index(self, index):
		res = []
		for turn_index in index:
			turn_trim = self.trim_index(turn_index)
			if turn_trim:
				res.append(turn_trim)
			else:
				break
		return res

	def multi_turn_index_to_sen(self, index, trim=True):
		if trim:
			index = self.multi_turn_trim_index(index)
		return list(map(lambda sent: \
							list(map(lambda word: self.vocab_list[word], sent)), \
						index))

	def index_to_sen(self, index, trim=True):
		if trim:
			index = self.trim_index(index)
		return list(map(lambda word: self.vocab_list[word], index))

	def get_sen(self, max_len, len, gen=False, pad=True):
		sen = []
		for i in range(len):
			sen.append(randrange(4, self.vocab_size))
		if not gen:
			sen[0] = self.vocab_to_index['<go>']
		sen[len - 1] = self.vocab_to_index['<eos>']
		if pad:
			for i in range(max_len - len):
				sen.append(self.vocab_to_index['<pad>'])
		return sen

	def get_data(self, reference_key=None, reference_len_key=None, gen_prob_key=None, gen_key=None, \
					   post_key=None, resp_key=None, context_key=None, multi_turn=False, to_list=False, \
				 pad=True, random_check=False, full_check=False, \
				 ref_len_flag=1, gen_len_flag=1, different_turn_len=False, \
				 batch=5, length=15):
		data = { \
			reference_key: [], \
			reference_len_key: [], \
			gen_prob_key: [], \
			gen_key: [], \
			post_key: [], \
			resp_key: [], \
			context_key: [], \
		}
		for i in range(batch):
			single_batch = { \
				reference_key: [], \
				reference_len_key: [], \
				gen_prob_key: [], \
				gen_key: [], \
				post_key: [], \
				resp_key: [], \
				context_key: [], \
			}
			turn_len = randrange(1, 5)
			for turn in range(turn_len if multi_turn else 1):
				ref = [[], [], [], []]
				gen = [[]]
				gen_prob = []
				ref_len = int()

				for j in range(4):
					tmp = randrange(2, length)
					ref[j] = self.get_sen(length, tmp, pad = True)
					if j == 0:
						ref_len = tmp

				gen[0] = self.get_sen(length, randrange(2, length), gen = True, pad = pad)

				if ref_len_flag < 1:
					ref[0] = self.get_sen(length, ref_len_flag + 2, gen = False, pad = True)
					ref_len = ref_len_flag
				if gen_len_flag < 1:
					gen[0] = self.get_sen(length, gen_len_flag + 1, gen = True, pad = True)

				for j in range(ref_len if not pad else length):
					vocab_prob = []
					for k in range(self.vocab_size):
						vocab_prob.append(random())
					vocab_prob /= np.sum(vocab_prob)
					if random_check == False:
						vocab_prob = np.log(vocab_prob)
					gen_prob.append(vocab_prob)

				if not multi_turn:
					if reference_key:
						data[reference_key].append(ref[0])
					if reference_len_key:
						data[reference_len_key].append(ref_len)
					if gen_prob_key:
						data[gen_prob_key].append(gen_prob)
					if gen_key:
						data[gen_key].append(gen[0])
					if post_key:
						data[post_key].append(ref[1])
					if resp_key:
						data[resp_key].append(ref[2])
					if context_key:
						data[context_key].append(ref[3])
				else:
					if reference_key and ((not different_turn_len) or randrange(0, 2) == 1):
						single_batch[reference_key].append(ref[0])
					if reference_len_key and ((not different_turn_len) or randrange(0, 2) == 1):
						single_batch[reference_len_key].append(ref_len)
					if gen_prob_key and ((not different_turn_len) or randrange(0, 2) == 1):
						single_batch[gen_prob_key].append(gen_prob)
					if gen_key and ((not different_turn_len) or randrange(0, 2) == 1):
						single_batch[gen_key].append(gen[0])
					if post_key and ((not different_turn_len) or randrange(0, 2) == 1):
						single_batch[post_key].append(ref[1])
					if resp_key and ((not different_turn_len) or randrange(0, 2) == 1):
						single_batch[resp_key].append(ref[2])
					if context_key and ((not different_turn_len) or randrange(0, 2) == 1):
						single_batch[context_key].append(ref[3])

			if multi_turn:
				for key in data.keys():
					data[key].append(single_batch[key])
		if full_check:
			if multi_turn:
				data[gen_prob_key][0][0][0] -= 1
			else:
				data[gen_prob_key][0][0] -= 1
		if not to_list:
			for i in data.keys():
				data[i] = np.array(data[i])
		return data


test_argument =  [ 'default',   'custom',   'custom',   'custom',   'custom',       'custom',     'custom',     'custom',     'custom',     'custom']
test_shape =     [     'pad',      'pad',      'jag',      'pad',      'pad',          'pad',        'pad',        'pad',        'pad',        'pad']
test_type =      [   'array',    'array',     'list',     'list',     'list',         'list',       'list',       'list',       'list',       'list']
test_batch_len = [   'equal',    'equal',    'equal',    'equal',  'unequal',        'equal',      'equal',      'equal',      'equal',      'equal']
test_turn_len =  [   'equal',    'equal',  'unequal',    'equal',    'equal',        'equal',      'equal',      'equal',      'equal',      'equal']
test_check =     ['no_check', 'no_check', 'no_check', 'no_check', 'no_check', 'random_check', 'full_check',   'no_check',   'no_check',   'no_check']
test_gen_len =   [         1,          1,          1,          1,          1,              1,            1,            0,            1,            0]
test_ref_len =   [         1,          1,          1,          1,          1,              1,            1,            1,            0,            0]



# test_argument =  [       'custom']
# test_shape =     [          'pad']
# test_type =      [         'list']
# test_batch_len = [        'unequal']
# test_turn_len =  [        'unequal']
# test_check =     [ 'random_check']
# test_gen_len =   [              1]
# test_ref_len =   [              1]

## test_batch_len: len(ref) == len(gen)?
## test_turn_len: len(single_batch(ref)) == len(single_batch(gen))?
## test_gen_len: 1 means normal, 0 means gen == empty
## test_ref_len: 1 means normal, 0 means ref == empty

perplexity_test_parameter = zip(test_argument, test_shape, test_type, \
							 test_batch_len, test_check)

def same_data(A, B):
	if type(A) != type(B):
		return False
	try:
		if len(A) != len(B):
			return False
	except TypeError:
		return A == B
	for i, x in enumerate(A):
		if not same_data(x, B[i]):
			return False
	return True

def same_dict(A, B):
	if A.keys() != B.keys():
		return False
	for x in A.keys():
		if not same_data(A[x], B[x]):
			return False
	return True

class TestPerlplexityMetric():
	def get_perplexity(self, input, reference_key='resp', reference_len_key='resp_length', \
					   gen_prob_key='gen_prob'):
		length_sum = 0
		word_loss = 0
		for i in range(len(input[reference_key])):
			max_length = input[reference_len_key][i]

			length_sum += max_length - 1
			for j in range(max_length - 1):
				word_loss += -(input[gen_prob_key][i][j][input[reference_key][i][j + 1]])
		return np.exp(word_loss / length_sum)

	@pytest.mark.parametrize('argument, shape, type, batch_len, check', perplexity_test_parameter)
	def test_close(self, argument, shape, type, batch_len, check):
		# 'default' or 'custom'
		# 'pad' or 'jag'
		# 'list' or 'array'
		# 'equal' or 'unequal'
		# 'random_check' or 'full_check' or 'no_check'
		dataloader = FakeDataLoader()
		reference_key, reference_len_key, gen_prob_key = ('resp', 'resp_length', 'gen_prob') \
			if argument == 'default' else ('rpk', 'rl', 'gp')
		random_check = check == 'random_check'
		full_check = check == 'full_check'
		data = dataloader.get_data(reference_key=reference_key, reference_len_key=reference_len_key, gen_prob_key=gen_prob_key, \
								   to_list=(type == 'list'), pad=(shape == 'pad'), \
								   random_check=random_check, full_check=full_check)
		_data = copy.deepcopy(data)
		if argument == 'default':
			pm = PerlplexityMetric(dataloader, full_check=full_check)
		else:
			pm = PerlplexityMetric(dataloader, reference_key, reference_len_key, gen_prob_key, full_check=full_check)

		print(batch_len)
		if batch_len == 'unequal':
			data[reference_key] = np.delete(data[reference_key], 1, 0)
			_data = copy.deepcopy(data)
			with pytest.raises(ValueError, match='Batch num is not matched.'):
				pm.forward(data)
		elif check == 'no_check':
			pm.forward(data)
			assert np.isclose(pm.close()['perplexity'], self.get_perplexity(data, reference_key, reference_len_key, \
																			gen_prob_key))
		else:
			print('random_check, full_check', random_check, full_check)
			with pytest.raises(ValueError, \
							   match='data\[gen_prob_key\] must be processed after log_softmax.'):
				pm.forward(data)
		assert same_dict(data, _data)

multi_perplexity_test_parameter = zip(test_argument, test_shape, test_type, \
							 test_batch_len, test_turn_len, test_check)

class TestMultiTurnPerplexityMetric:
	def get_perplexity(self, input, reference_key='sent', reference_len_key='sent_length', gen_prob_key='gen_prob'):
		length_sum = 0
		word_loss = 0
		for turn in range(len(input[reference_key])):
			for i in range(len(input[reference_key][turn])):
				max_length = input[reference_len_key][turn][i]

				length_sum += max_length - 1
				for j in range(max_length - 1):
					print(turn, i, j)
					word_loss += -(input[gen_prob_key][turn][i][j][input[reference_key][turn][i][j + 1]])
		return np.exp(word_loss / length_sum)

	@pytest.mark.parametrize('argument, shape, type, batch_len, turn_len, check', multi_perplexity_test_parameter)
	def test_close(self, argument, shape, type, batch_len, turn_len, check):
		# 'default' or 'custom'
		# 'pad' or 'jag'
		# 'list' or 'array'
		# 'equal' or 'unequal'
		# 'equal' or 'unequal'
		# 'random_check' or 'full_check' or 'no_check'
		dataloader = FakeDataLoader()
		reference_key, reference_len_key, gen_prob_key = ('sent', 'sent_length', 'gen_prob') \
			if argument == 'default' else ('rpk', 'rl', 'gp')
		random_check = check == 'random_check'
		full_check = check == 'full_check'
		data = dataloader.get_data(reference_key=reference_key, reference_len_key=reference_len_key, gen_prob_key=gen_prob_key, \
								   to_list=(type == 'list'), pad=(shape == 'pad'), \
								   multi_turn=True, different_turn_len=(turn_len == 'unequal'), \
								   random_check=random_check, full_check=full_check)
		_data = copy.deepcopy(data)
		if argument == 'default':
			mtpm = MultiTurnPerplexityMetric(dataloader)
		else:
			mtpm = MultiTurnPerplexityMetric(dataloader, reference_key, reference_len_key, gen_prob_key, \
											 full_check=full_check)

		if batch_len == 'unequal' or turn_len == 'unequal':
			data[reference_key] = np.delete(data[reference_key], 1, 0)
			_data = copy.deepcopy(data)
			with pytest.raises(ValueError, match='Batch num is not matched.'):
				mtpm.forward(data)
		elif check == 'no_check':
			mtpm.forward(data)
			assert np.isclose(mtpm.close()['perplexity'], self.get_perplexity(data, reference_key, reference_len_key, \
																			gen_prob_key))
		else:
			print('random_check, full_check', random_check, full_check)
			with pytest.raises(ValueError, \
							   match='data\[gen_prob_key\] must be processed after log_softmax.'):
				mtpm.forward(data)
		print(data)
		print(_data)
		assert same_dict(data, _data)


bleu_test_parameter = zip(test_argument, test_argument, test_type, test_batch_len, test_gen_len, test_ref_len)


class TestBleuCorpusMetric:
	def get_bleu(self, dataloader, input, reference_key, gen_key):
		refs = []
		gens = []
		for gen_sen, resp_sen in zip(input[gen_key], input[reference_key]):
			gen_sen_processed = dataloader.trim_index(gen_sen)
			resp_sen_processed = dataloader.trim_index(resp_sen[1:])
			refs.append([resp_sen_processed])
			gens.append(gen_sen_processed)
		print('refs:', refs)
		print('gens:', gens)
		print('bleu:', corpus_bleu(refs, gens, smoothing_function=SmoothingFunction().method7))
		return corpus_bleu(refs, gens, smoothing_function=SmoothingFunction().method7)

	@pytest.mark.parametrize('argument, shape, type, batch_len, gen_len, ref_len', bleu_test_parameter)
	def test_close(self, argument, shape, type, batch_len, gen_len, ref_len):
		# 'default' or 'custom'
		# 'pad' or 'jag'
		# 'list' or 'array'
		# 'equal' or 'unequal'
		# 0, 1
		# 0, 1
		dataloader = FakeDataLoader()
		reference_key, gen_key = ('resp', 'gen') \
			if argument == 'default' else ('rk', 'gpk')
		data = dataloader.get_data(reference_key=reference_key, gen_key=gen_key, \
								   to_list=(type == 'list'), pad=(shape == 'pad'), \
								   gen_len_flag=gen_len, ref_len_flag=ref_len)
		_data = copy.deepcopy(data)
		if argument == 'default':
			bcm = BleuCorpusMetric(dataloader)
		else:
			bcm = BleuCorpusMetric(dataloader, reference_key, gen_key)

		if batch_len == 'unequal':
			data[reference_key] = np.delete(data[reference_key], 1, 0)
			_data = copy.deepcopy(data)
			with pytest.raises(ValueError, match='Batch num is not matched.'):
				bcm.forward(data)
		else:
				bcm.forward(data)
				assert np.isclose(bcm.close()['bleu'], self.get_bleu(dataloader, data, reference_key, gen_key))
		assert same_dict(data, _data)

multi_bleu_test_parameter = zip(test_argument, test_shape, test_type, test_batch_len, \
								test_turn_len, test_gen_len, test_ref_len)


class TestMultiTurnBleuCorpusMetric:
	def get_bleu(self, dataloader, input, reference_key, gen_key):
		refs = []
		gens = []
		for i in range(len(input[reference_key])):
			for resp_sen, gen_sen in zip(input[reference_key][i], input[gen_key][i]):
				gen_sen_processed = dataloader.trim_index(gen_sen)
				resp_sen_processed = dataloader.trim_index(resp_sen)
				gens.append(gen_sen_processed)
				refs.append([resp_sen_processed[1:]])
		print('refs:', refs)
		print('gens:', gens)
		print('bleu:', corpus_bleu(refs, gens, smoothing_function=SmoothingFunction().method7))
		return corpus_bleu(refs, gens, smoothing_function=SmoothingFunction().method7)

	@pytest.mark.parametrize('argument, shape, type, batch_len, turn_len, gen_len, ref_len', multi_bleu_test_parameter)
	def test_close(self, argument, shape, type, batch_len, turn_len, gen_len, ref_len):
		# 'default' or 'custom'
		# 'pad' or 'jag'
		# 'list' or 'array'
		# 'equal' or 'unequal'
		# 'equal' or 'unequal'
		# 0, 1
		# 0, 1
		dataloader = FakeDataLoader()
		reference_key, gen_key = ('sent', 'gen') \
			if argument == 'default' else ('rk', 'gpk')
		data = dataloader.get_data(reference_key=reference_key, gen_key=gen_key, \
								   multi_turn=True, to_list=(type == 'list'), pad=(shape == 'pad'), \
								   gen_len_flag=gen_len, ref_len_flag=ref_len, different_turn_len=(turn_len == 'unequal'))
		_data = copy.deepcopy(data)
		if argument == 'default':
			mtbcm = MultiTurnBleuCorpusMetric(dataloader)
		else:
			mtbcm = MultiTurnBleuCorpusMetric(dataloader, reference_key, gen_key)

		if batch_len == 'unequal':
			data[reference_key] = np.delete(data[reference_key], 1, 0)
			_data = copy.deepcopy(data)
			with pytest.raises(ValueError, match='Batch num is not matched.'):
				mtbcm.forward(data)
		elif turn_len == 'unequal' or gen_len == 0:
			with pytest.raises(ValueError, match='Turn num is not matched.'):
				mtbcm.forward(data)
		else:
			mtbcm.forward(data)
			assert np.isclose(mtbcm.close()['bleu'], self.get_bleu(dataloader, data, reference_key, gen_key))
		assert same_dict(data, _data)


single_turn_dialog_recorder_test_parameter = zip(test_argument, test_shape, test_type, test_batch_len)


class TestSingleTurnDialogRecorder():
	def get_sen_from_index(self, dataloader, data, post_key='post', resp_key='resp', gen_key='gen'):
		ans = { \
			'post': [], \
			'resp': [], \
			'gen': [], \
			}
		for sen in data[post_key]:
			ans['post'].append(dataloader.index_to_sen(sen[1:]))
			print(ans['post'][-1])
		for sen in data[resp_key]:
			ans['resp'].append(dataloader.index_to_sen(sen[1:]))
			print(ans['resp'][-1])
		for sen in data[gen_key]:
			ans['gen'].append(dataloader.index_to_sen(sen))

		return ans

	@pytest.mark.parametrize('argument, shape, type, batch_len', single_turn_dialog_recorder_test_parameter)
	def test_close(self, argument, shape, type, batch_len):
		# 'default' or 'custom'
		# 'pad' or 'jag'
		# 'list' or 'array'
		# 'equal' or 'unequal'
		dataloader = FakeDataLoader()
		post_key, resp_key, gen_key = ('post', 'resp', 'gen') \
			if argument == 'default' else ('pk', 'rk', 'gk')
		data = dataloader.get_data(post_key=post_key, resp_key=resp_key, gen_key=gen_key, \
								   to_list=(type == 'list'), pad=(shape == 'pad'))
		_data = copy.deepcopy(data)
		if argument == 'default':
			sr = SingleTurnDialogRecorder(dataloader)
		else:
			sr = SingleTurnDialogRecorder(dataloader, post_key, resp_key, gen_key)

		if batch_len == 'unequal':
			data[post_key] = np.delete(data[post_key], 1, 0)
			_data = copy.deepcopy(data)
			with pytest.raises(ValueError, match='Batch num is not matched.'):
				sr.forward(data)
		else:
			sr.forward(data)
			assert sr.close() == self.get_sen_from_index(dataloader, data, post_key, resp_key, \
																			gen_key)
		assert same_dict(data, _data)


multi_turn_dialog_test_parameter = zip(test_argument, test_shape, test_type, \
									   test_batch_len, test_gen_len)


class TestMultiTurnDialogRecorder:
	def get_sen_from_index(self, dataloader, data, post_key='context', resp_key='reference', gen_key='gen'):
		ans = { \
			'context': [], \
			'reference': [], \
			'gen': [], \
			}
		for turn in data[post_key]:
			ans['context'].append(dataloader.multi_turn_index_to_sen(np.array(turn)[ :, 1 :]))
		for turn in data[resp_key]:
			ans['reference'].append(dataloader.multi_turn_index_to_sen(np.array(turn)[ :, 1 :]))
		for turn in data[gen_key]:
			ans['gen'].append(dataloader.multi_turn_index_to_sen(turn))

		return ans

	@pytest.mark.parametrize('argument, shape, type, batch_len, gen_len', multi_turn_dialog_test_parameter)
	def test_close(self, argument, shape, type, batch_len, gen_len):
		# 'default' or 'custom'
		# 'pad' or 'jag'
		# 'list' or 'array'
		# 'equal' or 'unequal'
		# 'equal' or 'unequal'
		# 0, 1
		dataloader = FakeDataLoader()
		context_key, reference_key, gen_key = ('context', 'reference', 'gen') \
			if argument == 'default' else ('ck', 'rk', 'gk')
		data = dataloader.get_data(context_key=context_key, reference_key=reference_key, gen_key=gen_key, \
								   multi_turn=True, to_list=(type == 'list'), pad=(shape == 'pad'), \
								   gen_len_flag=gen_len)
		_data = copy.deepcopy(data)
		if argument == 'default':
			mtbr = MultiTurnDialogRecorder(dataloader)
		else:
			mtbr = MultiTurnDialogRecorder(dataloader, context_key, reference_key, gen_key)

		if batch_len == 'unequal':
			data[reference_key] = np.delete(data[reference_key], 1, 0)
			_data = copy.deepcopy(data)
			with pytest.raises(ValueError, match='Batch num is not matched.'):
				mtbr.forward(data)
		else:
			mtbr.forward(data)
			assert mtbr.close() == self.get_sen_from_index(dataloader, data, context_key, reference_key, gen_key)
		assert same_dict(data, _data)


language_generation_test_parameter = zip(test_argument, test_shape, test_type)


class TestLanguageGenerationRecorder():
	def get_sen_from_index(self, dataloader, data, gen_key = 'gen'):
		ans = []
		for sen in data[gen_key]:
			ans.append(dataloader.index_to_sen(sen))
			print(ans[-1])
		return ans

	@pytest.mark.parametrize('argument, shape, type', language_generation_test_parameter)
	def test_close(self, argument, shape, type):
		# 'default' or 'custom'
		# 'pad' or 'jag'
		# 'list' or 'array'
		# 'equal' or 'unequal'
		dataloader = FakeDataLoader()
		gen_key = 'gen' \
			if argument == 'default' else 'gk'
		data = dataloader.get_data(gen_key=gen_key, \
								   to_list=(type == 'list'), pad=(shape == 'pad'))
		_data = copy.deepcopy(data)
		if argument == 'default':
			lg = LanguageGenerationRecorder(dataloader)
		else:
			lg = LanguageGenerationRecorder(dataloader, gen_key)

		lg.forward(data)
		assert lg.close()['gen'] == self.get_sen_from_index(dataloader, data, gen_key)
		assert same_dict(data, _data)

class TestMetricChain():
	def test_init(self):
		mc = MetricChain()

	def test_add_metric(self):
		mc = MetricChain()
		with pytest.raises(TypeError):
			mc.add_metric([1, 2, 3])

	def test_close1(self):
		dataloader = FakeDataLoader()
		data = dataloader.get_data(reference_key='reference_key', reference_len_key='reference_len_key', gen_prob_key='gen_prob_key', \
								   gen_key='gen_key', post_key='post_key', resp_key='resp_key', \
								   multi_turn=True)
		pm = MultiTurnPerplexityMetric(dataloader, 'reference_key', 'reference_len_key', 'gen_prob_key',
									   full_check=True)
		perplexity = TestMultiTurnPerplexityMetric().get_perplexity(data, 'reference_key', 'reference_len_key', 'gen_prob_key')

		bcm = MultiTurnBleuCorpusMetric(dataloader, 'reference_key', 'gen_key')
		bleu = TestMultiTurnBleuCorpusMetric().get_bleu(dataloader, data, 'reference_key', 'gen_key')

		_data = copy.deepcopy(data)
		mc = MetricChain()
		mc.add_metric(pm)
		mc.add_metric(bcm)
		mc.forward(data)
		res = mc.close()

		assert np.isclose(res['perplexity'], perplexity)
		assert np.isclose(res['bleu'], bleu)
		print(data)
		print(_data)
		assert same_dict(data, _data)
