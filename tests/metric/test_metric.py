import copy
import itertools

import numpy as np
import pytest

from random import random, randrange
from contk.metric import MetricBase, PerlplexityMetric, MultiTurnPerplexityMetric, BleuCorpusMetric, \
	MultiTurnBleuCorpusMetric, SingleTurnDialogRecorder, MultiTurnDialogRecorder, LanguageGenerationRecorder, \
	MetricChain
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
from contk.dataloader import BasicLanguageGeneration, MultiTurnDialog

def test_bleu_bug():
	ref = [[[1, 3], [3], [4]]]
	gen = [[1]]
	with pytest.raises(ZeroDivisionError):
		corpus_bleu(ref, gen, smoothing_function=SmoothingFunction().method7)


class FakeDataLoader(BasicLanguageGeneration):
	def __init__(self):
		self.all_vocab_list  = ['<pad>', '<unk>', '<go>', '<eos>', 'what', 'how', 'here', 'do']
		self.valid_vocab_len = 8
		self.vocab_to_index = {x: i for i, x in enumerate(self.vocab_list)}
		self.end_token = 3
		self.pad_id = 0

	def get_sen(self, max_len, len, gen=False, pad=True, end_token="<eos>"):
		sen = []
		for i in range(len):
			sen.append(randrange(self.vocab_to_index['<eos>'] + 1, self.vocab_size))
		if not gen:
			sen[0] = self.vocab_to_index['<go>']
		sen[len - 1] = self.vocab_to_index['<eos>']
		if pad:
			for i in range(max_len - len):
				sen.append(self.vocab_to_index['<pad>'])
		return sen

	def get_data(self, reference_key=None, reference_len_key=None, gen_prob_key=None, gen_key=None, \
				 post_key=None, \
				 to_list=False, \
				 pad=True, gen_prob_check='no_check', \
				 gen_len='random', ref_len='random', \
				 batch=5, max_len=10):
		data = { \
			reference_key: [], \
			reference_len_key: [], \
			gen_prob_key: [], \
			gen_key: [], \
			post_key: [] \
		}

		for i in range(batch):
			if ref_len == "random":
				ref_nowlen = randrange(2, 5)
			elif ref_len == "non-empty":
				ref_nowlen = 8
			elif ref_len == 'empty':
				ref_nowlen = 2
			data[reference_key].append(self.get_sen(max_len, ref_nowlen, pad=pad))
			data[reference_len_key].append(ref_nowlen)

			data[post_key].append(self.get_sen(max_len, ref_nowlen, pad=pad))

			if gen_len == "random":
				gen_nowlen = randrange(1, 4) if i != 0 else 3 # for BLEU not empty
			elif gen_len == "non-empty":
				gen_nowlen = 7
			elif gen_len == "empty":
				gen_nowlen = 1
			data[gen_key].append(self.get_sen(max_len, gen_nowlen, gen=True, pad=pad))

			gen_prob = []
			for j in range(max_len if pad else ref_nowlen):
				vocab_prob = []
				for k in range(self.vocab_size):
					vocab_prob.append(random())
				vocab_prob /= np.sum(vocab_prob)
				if gen_prob_check != "random_check":
					vocab_prob = np.log(vocab_prob)
				gen_prob.append(list(vocab_prob))
			data[gen_prob_key].append(gen_prob)

		if gen_prob_check == "full_check":
			data[gen_prob_key][0][0][0] -= 1

		if not to_list:
			for key in data:
				if key is not None:
					data[key] = np.array(data[key])
		return data

class FakeMultiDataloader(MultiTurnDialog):
	def __init__(self):
		self.all_vocab_list  = ['<pad>', '<unk>', '<go>', '<eos>', 'what', 'how', 'here', 'do']
		self.valid_vocab_len = 8
		self.vocab_to_index = {x: i for i, x in enumerate(self.vocab_list)}
		self.end_token = 4
		self.pad_id = 0

	def get_sen(self, max_len, len, gen=False, pad=True):
		return FakeDataLoader.get_sen(self, max_len, len, gen, pad, end_token="<eot>")

	def get_data(self, reference_key=None, reference_len_key=None, turn_len_key=None, gen_prob_key=None, gen_key=None, \
			context_key=None, \
			to_list=False, \
			pad=True, gen_prob_check='no_check', \
			gen_len='random', ref_len='random', \
			batch=5, max_len=10, max_turn=5):
		data = { \
			reference_key: [], \
			reference_len_key: [], \
			turn_len_key: [], \
			gen_prob_key: [], \
			gen_key: [], \
			context_key: [] \
		}
		
		for i in range(batch):
			turn_length = randrange(1, max_turn+1)
			turn_reference = []
			turn_reference_len = []
			turn_gen_prob = []
			turn_gen = []
			turn_context = []

			for j in range(turn_length):
				if ref_len == "random":
					ref_nowlen = randrange(2, 5)
				elif ref_len == "non-empty":
					ref_nowlen = 8
				elif ref_len == 'empty':
					ref_nowlen = 2
				turn_reference.append(self.get_sen(max_len, ref_nowlen, pad=pad))
				turn_reference_len.append(ref_nowlen)

				turn_context.append(self.get_sen(max_len, ref_nowlen, pad=pad))
				if gen_len == "random":
					gen_nowlen = randrange(1, 4) if i != 0 else 3 # for BLEU not empty
				elif gen_len == "non-empty":
					gen_nowlen = 7
				elif gen_len == "empty":
					gen_nowlen = 1
				turn_gen.append(self.get_sen(max_len, gen_nowlen, gen=True, pad=pad))

				gen_prob = []
				for k in range(max_len if pad else ref_nowlen):
					vocab_prob = []
					for l in range(self.vocab_size):
						vocab_prob.append(random())
					vocab_prob /= np.sum(vocab_prob)
					if gen_prob_check != "random_check":
						vocab_prob = np.log(vocab_prob)
					gen_prob.append(list(vocab_prob))
				turn_gen_prob.append(gen_prob)

			data[reference_key].append(turn_reference)
			data[reference_len_key].append(turn_reference_len)
			data[turn_len_key].append(turn_length)
			data[gen_prob_key].append(turn_gen_prob)
			data[gen_key].append(turn_gen)
			data[context_key].append(turn_context)

		if gen_prob_check == "full_check":
			data[gen_prob_key][0][0][0][0] -= 1

		if not to_list:
			for key in data:
				if key is not None:
					data[key] = np.array(data[key])
		return data

test_argument =  [ 'default',   'custom']

test_shape =     [     'pad',      'jag',      'pad',      'jag']
test_type =      [   'array',    'array',     'list',     'list']

test_batch_len = [   'equal',  'unequal']
#test_turn_len =  [   'equal',  'unequal']

test_check =     ['no_check', 'random_check', 'full_check']

test_gen_len =   [  'random', 'non-empty',   'empty']
test_ref_len =   [  'random', 'non-empty',   'empty']

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
## test_gen_len: 'empty' means all length are 1 (eos), 'non-empty' means all length are > 1, 'random' means length are random
## test_ref_len: 'empty' means all length are 2 (eos), 'non-empty' means all length are > 2, 'both' means length are random

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

def generate_testcase(*args):
	args = [(list(p), mode) for p, mode in args]
	default = []
	for p, _ in args:
		default.extend(p[0])
	#yield tuple(default)
	# add
	i = 0
	for p, mode in args:
		if mode == "add":
			for k in p[1:]:
				yield tuple(default[:i] + list(k) + default[i+1:])
		i += len(p[0])

	# multi
	res = []
	for i, (p, mode) in enumerate(args):
		if mode == "add":
			res.append(p[:1])
		else:
			res.append(p)
	for p in itertools.product(*res):
		yield tuple(itertools.chain(*p))

perplexity_test_parameter = generate_testcase(\
	(zip(test_argument), "add"),
	(zip(test_shape, test_type), "multi"),
	(zip(test_batch_len), "add"),
	(zip(test_check), "add"),
	(zip(test_ref_len), "multi")
)

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

	@pytest.mark.parametrize('argument, shape, type, batch_len, check, ref_len', perplexity_test_parameter)
	def test_close(self, argument, shape, type, batch_len, check, ref_len):
		# 'default' or 'custom'
		# 'pad' or 'jag'
		# 'list' or 'array'
		# 'equal' or 'unequal'
		# 'random_check' or 'full_check' or 'no_check'
		# 'random', 'non-empty', 'empty'
		dataloader = FakeDataLoader()
		reference_key, reference_len_key, gen_prob_key = ('resp', 'resp_length', 'gen_prob') \
			if argument == 'default' else ('rpk', 'rl', 'gp')
		data = dataloader.get_data(reference_key=reference_key, reference_len_key=reference_len_key, gen_prob_key=gen_prob_key, \
								   to_list=(type == 'list'), pad=(shape == 'pad'), \
								   gen_prob_check=check, ref_len=ref_len)
		_data = copy.deepcopy(data)
		if argument == 'default':
			pm = PerlplexityMetric(dataloader, full_check=(check=='full_check'))
		else:
			pm = PerlplexityMetric(dataloader, reference_key, reference_len_key, gen_prob_key, full_check=(check=='full_check'))

		print(batch_len)
		if batch_len == 'unequal':
			data[reference_key] = data[reference_key][1:]
			_data = copy.deepcopy(data)
			with pytest.raises(ValueError, match='Batch num is not matched.'):
				pm.forward(data)
		elif check == 'no_check':
			pm.forward(data)
			assert np.isclose(pm.close()['perplexity'], self.get_perplexity(data, reference_key, reference_len_key, \
																			gen_prob_key))
		else:
			with pytest.raises(ValueError, \
							   match='data\[gen_prob_key\] must be processed after log_softmax.'):
				pm.forward(data)
		assert same_dict(data, _data)

multi_perplexity_test_parameter = generate_testcase(\
	(zip(test_argument), "add"),
	(zip(test_shape, test_type), "multi"),
	(zip(test_batch_len), "add"),
	(zip(test_check), "add"),
	(zip(test_ref_len), "multi")
)

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

	@pytest.mark.parametrize('argument, shape, type, batch_len, check, ref_len', multi_perplexity_test_parameter)
	def test_close(self, argument, shape, type, batch_len, check, ref_len):
		# 'default' or 'custom'
		# 'pad' or 'jag'
		# 'list' or 'array'
		# 'equal' or 'unequal'
		# 'equal' or 'unequal'
		# 'random_check' or 'full_check' or 'no_check'
		# 'random', 'non-empty', 'empty'
		dataloader = FakeMultiDataloader()
		reference_key, reference_len_key, gen_prob_key = ('sent', 'sent_length', 'gen_prob') \
			if argument == 'default' else ('rpk', 'rl', 'gp')
		data = dataloader.get_data(reference_key=reference_key, reference_len_key=reference_len_key, gen_prob_key=gen_prob_key, \
								   to_list=(type == 'list'), pad=(shape == 'pad'), \
								   gen_prob_check=check, ref_len=ref_len)
		_data = copy.deepcopy(data)
		if argument == 'default':
			mtpm = MultiTurnPerplexityMetric(dataloader, full_check=(check=="full_check"))
		else:
			mtpm = MultiTurnPerplexityMetric(dataloader, reference_key, reference_len_key, gen_prob_key, \
											 full_check=(check=="full_check"))

		if batch_len == 'unequal':
			data[reference_key] = data[reference_key][1:]
			_data = copy.deepcopy(data)
			with pytest.raises(ValueError, match='Batch num is not matched.'):
				mtpm.forward(data)
		elif check == 'no_check':
			mtpm.forward(data)
			assert np.isclose(mtpm.close()['perplexity'], self.get_perplexity(data, reference_key, reference_len_key, \
																			gen_prob_key))
		else:
			with pytest.raises(ValueError, \
							   match='data\[gen_prob_key\] must be processed after log_softmax.'):
				mtpm.forward(data)
		print(data)
		print(_data)
		assert same_dict(data, _data)


bleu_test_parameter = generate_testcase(\
	(zip(test_argument), "add"),
	(zip(test_shape, test_type), "multi"),
	(zip(test_batch_len), "add"),
	(zip(test_gen_len), "multi"),
	(zip(test_ref_len), "multi")
)

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
		# 'random', 'non-empty', 'empty'
		# 'random', 'non-empty', 'empty'
		dataloader = FakeDataLoader()
		reference_key, gen_key = ('resp', 'gen') \
			if argument == 'default' else ('rk', 'gpk')
		data = dataloader.get_data(reference_key=reference_key, gen_key=gen_key, \
								   to_list=(type == 'list'), pad=(shape == 'pad'), \
								   gen_len=gen_len, ref_len=ref_len)
		_data = copy.deepcopy(data)
		if argument == 'default':
			bcm = BleuCorpusMetric(dataloader)
		else:
			bcm = BleuCorpusMetric(dataloader, reference_key, gen_key)

		if batch_len == 'unequal':
			data[reference_key] = data[reference_key][1:]
			_data = copy.deepcopy(data)
			with pytest.raises(ValueError, match='Batch num is not matched.'):
				bcm.forward(data)
		else:
				bcm.forward(data)
				assert np.isclose(bcm.close()['bleu'], self.get_bleu(dataloader, data, reference_key, gen_key))
		assert same_dict(data, _data)

multi_bleu_test_parameter = generate_testcase(\
	(zip(test_argument), "add"),
	(zip(test_shape, test_type), "multi"),
	(zip(test_batch_len), "add"),
	(zip(test_gen_len), "multi"),
	(zip(test_ref_len), "multi")
)


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

	@pytest.mark.parametrize('argument, shape, type, batch_len, gen_len, ref_len', multi_bleu_test_parameter)
	def test_close(self, argument, shape, type, batch_len, gen_len, ref_len):
		# 'default' or 'custom'
		# 'pad' or 'jag'
		# 'list' or 'array'
		# 'equal' or 'unequal'
		# 'random', 'non-empty', 'empty'
		# 'random', 'non-empty', 'empty'
		dataloader = FakeMultiDataloader()
		reference_key, turn_len_key, gen_key = ('reference', 'turn_length', 'gen') \
			if argument == 'default' else ('rk', 'tlk', 'gpk')
		data = dataloader.get_data(reference_key=reference_key, turn_len_key=turn_len_key, gen_key=gen_key, \
								   to_list=(type == 'list'), pad=(shape == 'pad'), \
								   gen_len=gen_len, ref_len=ref_len)
		_data = copy.deepcopy(data)
		if argument == 'default':
			mtbcm = MultiTurnBleuCorpusMetric(dataloader)
		else:
			mtbcm = MultiTurnBleuCorpusMetric(dataloader, reference_key, gen_key, turn_len_key)

		if batch_len == 'unequal':
			data[reference_key] = data[reference_key][1:]
			_data = copy.deepcopy(data)
			with pytest.raises(ValueError, match='Batch num is not matched.'):
				mtbcm.forward(data)
		else:
			mtbcm.forward(data)
			assert np.isclose(mtbcm.close()['bleu'], self.get_bleu(dataloader, data, reference_key, gen_key))
		assert same_dict(data, _data)


single_turn_dialog_recorder_test_parameter = generate_testcase(\
	(zip(test_argument), "add"),
	(zip(test_shape, test_type), "multi"),
	(zip(test_batch_len), "add"),
	(zip(test_gen_len), "multi"),
	(zip(test_ref_len), "multi")
)

class TestSingleTurnDialogRecorder():
	def get_sen_from_index(self, dataloader, data, post_key='post', reference_key='resp', gen_key='gen'):
		ans = { \
			'post': [], \
			'resp': [], \
			'gen': [], \
			}
		for sen in data[post_key]:
			ans['post'].append(dataloader.index_to_sen(sen[1:]))
			print(ans['post'][-1])
		for sen in data[reference_key]:
			ans['resp'].append(dataloader.index_to_sen(sen[1:]))
			print(ans['resp'][-1])
		for sen in data[gen_key]:
			ans['gen'].append(dataloader.index_to_sen(sen))

		return ans

	@pytest.mark.parametrize('argument, shape, type, batch_len, gen_len, ref_len', single_turn_dialog_recorder_test_parameter)
	def test_close(self, argument, shape, type, batch_len, gen_len, ref_len):
		# 'default' or 'custom'
		# 'pad' or 'jag'
		# 'list' or 'array'
		# 'equal' or 'unequal'
		dataloader = FakeDataLoader()
		post_key, reference_key, gen_key = ('post', 'resp', 'gen') \
			if argument == 'default' else ('pk', 'rk', 'gk')
		data = dataloader.get_data(post_key=post_key, reference_key=reference_key, gen_key=gen_key, \
								   to_list=(type == 'list'), pad=(shape == 'pad'),
								   gen_len=gen_len, ref_len=ref_len)
		_data = copy.deepcopy(data)
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
	(zip(test_argument), "add"),
	(zip(test_shape, test_type), "multi"),
	(zip(test_batch_len), "add"),
	(zip(test_gen_len), "multi"),
	(zip(test_ref_len), "multi")
)

class TestMultiTurnDialogRecorder:
	def check(self, ans, dataloader, data, post_key='context', resp_key='reference', gen_key='gen', turn_length='turn_length'):
		assert len(ans['context']) == len(ans['reference'])
		for i, turn in enumerate(data[turn_length]):
			assert len(ans['reference'][i]) == turn
			assert len(ans['gen'][i]) == turn

	@pytest.mark.parametrize('argument, shape, type, batch_len, gen_len, ref_len', multi_turn_dialog_test_parameter)
	def test_close(self, argument, shape, type, batch_len, gen_len, ref_len):
		# 'default' or 'custom'
		# 'pad' or 'jag'
		# 'list' or 'array'
		# 'equal' or 'unequal'
		# 'random', 'non-empty', 'empty'
		# 'random', 'non-empty', 'empty'
		dataloader = FakeMultiDataloader()
		context_key, reference_key, gen_key, turn_len_key = ('context', 'reference', 'gen', 'turn_length') \
			if argument == 'default' else ('ck', 'rk', 'gk', 'tk')
		data = dataloader.get_data(context_key=context_key, turn_len_key=turn_len_key, reference_key=reference_key, gen_key=gen_key, \
								   to_list=(type == 'list'), pad=(shape == 'pad'), \
								   gen_len=gen_len, ref_len=ref_len)
		_data = copy.deepcopy(data)
		if argument == 'default':
			mtbr = MultiTurnDialogRecorder(dataloader)
		else:
			mtbr = MultiTurnDialogRecorder(dataloader, context_key, reference_key, gen_key, turn_len_key)

		if batch_len == 'unequal':
			data[reference_key] = data[reference_key][1:]
			_data = copy.deepcopy(data)
			with pytest.raises(ValueError, match='Batch num is not matched.'):
				mtbr.forward(data)
		else:
			mtbr.forward(data)
			self.check(mtbr.close(), dataloader, \
				data, context_key, reference_key, gen_key, turn_len_key)

		assert same_dict(data, _data)

language_generation_test_parameter = generate_testcase(\
	(zip(test_argument), "add"),
	(zip(test_shape, test_type), "multi"),
	(zip(test_gen_len), "multi"),
)


class TestLanguageGenerationRecorder():
	def get_sen_from_index(self, dataloader, data, gen_key='gen'):
		ans = []
		for sen in data[gen_key]:
			ans.append(dataloader.index_to_sen(sen))
			print(ans[-1])
		return ans

	@pytest.mark.parametrize('argument, shape, type, gen_len', language_generation_test_parameter)
	def test_close(self, argument, shape, type, gen_len):
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
		dataloader = FakeMultiDataloader()
		data = dataloader.get_data(reference_key='reference_key', reference_len_key='reference_len_key', turn_len_key='turn_len_key', gen_prob_key='gen_prob_key', \
								   gen_key='gen_key', context_key='context_key')
		pm = MultiTurnPerplexityMetric(dataloader, 'reference_key', 'reference_len_key', 'gen_prob_key',
									   full_check=True)
		perplexity = TestMultiTurnPerplexityMetric().get_perplexity(data, 'reference_key', 'reference_len_key', 'gen_prob_key')

		bcm = MultiTurnBleuCorpusMetric(dataloader, 'reference_key', 'gen_key', 'turn_len_key')
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
