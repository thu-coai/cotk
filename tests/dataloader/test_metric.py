import copy

import numpy as np
import pytest

from random import random, randrange
from contk.metric import MetricBase, PerlplexityMetric, MultiTurnPerplexityMetric, BleuCorpusMetric, \
	MultiTurnBleuCorpusMetric, SingleTurnDialogRecorder, MultiTurnDialogRecorder, LanguageGenerationRecorder, \
	MetricChain
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction


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

	def trim_index(self, index):
		index = self.trim_before_target(list(index), 3)
		idx = len(index)
		while index[idx - 1] == 0:
			idx -= 1
		index = index[:idx]
		return index

	def index_to_sen(self, index, trim=True):
		if trim:
			index = self.trim_index(index)
		return list(map(lambda word: self.vocab_list[word], index))

	def get_data(self, reference_key, reference_len_key, gen_prob_key, gen_key, post_key, resp_key, res_key,\
				 log_softmax = True, to_list = False, multi_turn = False):
		data = {reference_key: [], reference_len_key: [], gen_prob_key: [],\
				gen_key: [], post_key: [], resp_key: [], res_key: []}
		batch = 5
		length = 15
		for i in range(batch):
			single_batch = {reference_key: [], reference_len_key: [], gen_prob_key: [], \
						   gen_key: [], post_key: [], resp_key: [], res_key: []}
			for turn in range(5 if multi_turn else 1):
				sen_str = []
				sen_int = []
				res_str = []
				res_int = []
				sen_prob = []
				for j in range(length):
					t = randrange(self.vocab_size)
					if t == 3 or t == 2:
						t = 5
					sen_str.append(self.vocab_list[t])
					sen_int.append(t)

					t = randrange(self.vocab_size)
					if t == 3 or t == 2:
						t = 5
					res_str.append(self.vocab_list[t])
					res_int.append(t)

					vocab_prob = []
					for k in range(self.vocab_size):
						vocab_prob.append(random())
					vocab_prob /= np.sum(vocab_prob)
					if log_softmax:
						vocab_prob = np.log(vocab_prob)
					sen_prob.append(vocab_prob)

				sen_str[0] = '<go>'
				sen_str[-1] = '<eos>'
				res_str[-1] = '<eos>'
				sen_int[0] = self.vocab_to_index['<go>']
				sen_int[-1] = self.vocab_to_index['<eos>']
				res_int[-1] = self.vocab_to_index['<eos>']

				if not multi_turn:
					data[reference_key].append(sen_int)
					data[reference_len_key].append(length)
					data[gen_key].append(res_str)
					data[gen_prob_key].append(sen_prob)
					data[post_key].append(sen_int)
					data[resp_key].append(res_int)
					data[res_key].append(res_int)
				else:
					single_batch[reference_key].append(sen_int)
					single_batch[reference_len_key].append(length)
					single_batch[gen_key].append(res_str)
					single_batch[gen_prob_key].append(sen_prob)
					single_batch[post_key].append(sen_int)
					single_batch[resp_key].append(res_int)
					single_batch[res_key].append(res_int)

			if multi_turn:
				for key in data.keys():
					data[key].append(single_batch[key])

		if not to_list:
			for i in data.keys():
				data[i] = np.array(data[i])
		return data



class TestPerlplexityMetric():
	def get_perplexity(self, input):
		length_sum = 0
		word_loss = 0
		for i in range(len(input['reference_key'])):
			max_length = input['reference_len_key'][i]

			length_sum += max_length - 1
			for j in range(max_length - 1):
				# word_loss += -(input['gen_prob_key'][i][j][ FakeDataLoader().vocab_to_index[input['reference_key'][i][j + 1]] ])
				word_loss += -(input['gen_prob_key'][i][j][ input['reference_key'][i][j + 1] ])
		return np.exp(word_loss / length_sum)

	def test_close1(self):
		dataloader = FakeDataLoader()
		data = dataloader.get_data('reference_key', 'reference_len_key', 'gen_prob_key',\
								   'gen_key', 'post_key', 'resp_key', 'res_key', \
								   True, False)
		ans = self.get_perplexity(data)

		pm = PerlplexityMetric(dataloader, 'reference_key', 'reference_len_key', 'gen_prob_key', full_check=True)
		pm.forward(data)
		assert np.isclose(pm.close()['perplexity'], ans)

	def test_close2(self):
		dataloader = FakeDataLoader()
		data = dataloader.get_data('reference_key', 'reference_len_key', 'gen_prob_key',\
								   'gen_key', 'post_key', 'resp_key', 'res_key', \
								   True, False)

		data['reference_key'] = np.delete(data['reference_key'], 1)

		pm = PerlplexityMetric(dataloader, 'reference_key', 'reference_len_key', 'gen_prob_key', full_check=True)

		with pytest.raises(ValueError):
			pm.forward(data)

	def test_close3(self):
		dataloader = FakeDataLoader()
		data = dataloader.get_data('reference_key', 'reference_len_key', 'gen_prob_key',\
								   'gen_key', 'post_key', 'resp_key', 'res_key', \
								   False, True)
		ans = self.get_perplexity(data)

		pm = PerlplexityMetric(dataloader, 'reference_key', 'reference_len_key', 'gen_prob_key', full_check=True)
		with pytest.raises(ValueError):
			pm.forward(data)

	def test_close4(self):
		dataloader = FakeDataLoader()
		data = dataloader.get_data('reference_key', 'reference_len_key', 'gen_prob_key',\
								   'gen_key', 'post_key', 'resp_key', 'res_key', \
								   True, True)
		ans = self.get_perplexity(data)

		pm = PerlplexityMetric(dataloader, 'reference_key', 'reference_len_key', 'gen_prob_key', full_check=True)
		pm.forward(data)
		assert np.isclose(pm.close()['perplexity'], ans)


	def test_close5(self):
		dataloader = FakeDataLoader()
		data = dataloader.get_data('reference_key', 'reference_len_key', 'gen_prob_key',\
								   'gen_key', 'post_key', 'resp_key', 'res_key', \
								   log_softmax=False, to_list=True)

		pm = PerlplexityMetric(dataloader, 'reference_key', 'reference_len_key', 'gen_prob_key', full_check=True)
		with pytest.raises(ValueError):
			pm.forward(data)

class TestMultiTurnPerplexityMetric:
	def get_perplexity(self, input):
		length_sum = 0
		word_loss = 0
		for turn in range(len(input['reference_key'])):
			for i in range(len(input['reference_key'][turn])):
				max_length = input['reference_len_key'][turn][i]

				length_sum += max_length - 1
				for j in range(max_length - 1):
					# word_loss += -(input['gen_prob_key'][i][j][ FakeDataLoader().vocab_to_index[input['reference_key'][i][j + 1]] ])
					word_loss += -(input['gen_prob_key'][turn][i][j][ input['reference_key'][turn][i][j + 1] ])
		return np.exp(word_loss / length_sum)

	def test_close1(self):
		dataloader = FakeDataLoader()
		data = dataloader.get_data('reference_key', 'reference_len_key', 'gen_prob_key',\
								   'gen_key', 'post_key', 'resp_key', 'res_key', \
								   True, False, True)

		ans = self.get_perplexity(data)

		pm = MultiTurnPerplexityMetric(dataloader, 'reference_key', 'reference_len_key', 'gen_prob_key', full_check=True)
		pm.forward(data)
		assert np.isclose(pm.close()['perplexity'], ans)

	def test_close2(self):
		dataloader = FakeDataLoader()
		data = dataloader.get_data('reference_key', 'reference_len_key', 'gen_prob_key',\
								   'gen_key', 'post_key', 'resp_key', 'res_key', \
								   True, False, True)

		data['reference_key'] = np.delete(data['reference_key'], 1)

		pm = MultiTurnPerplexityMetric(dataloader, 'reference_key', 'reference_len_key', 'gen_prob_key', full_check=True)

		with pytest.raises(ValueError):
			pm.forward(data)

	def test_close3(self):
		dataloader = FakeDataLoader()
		data = dataloader.get_data('reference_key', 'reference_len_key', 'gen_prob_key',\
								   'gen_key', 'post_key', 'resp_key', 'res_key', \
								   False, True, True)
		ans = self.get_perplexity(data)

		pm = MultiTurnPerplexityMetric(dataloader, 'reference_key', 'reference_len_key', 'gen_prob_key', full_check=True)
		with pytest.raises(ValueError):
			pm.forward(data)

	def test_close4(self):
		dataloader = FakeDataLoader()
		data = dataloader.get_data('reference_key', 'reference_len_key', 'gen_prob_key',\
								   'gen_key', 'post_key', 'resp_key', 'res_key', \
								   True, True, True)
		ans = self.get_perplexity(data)

		pm = MultiTurnPerplexityMetric(dataloader, 'reference_key', 'reference_len_key', 'gen_prob_key', full_check=True)
		pm.forward(data)
		assert np.isclose(pm.close()['perplexity'], ans)

class TestBleuCorpusMetric:
	def get_bleu(self, dataloader, input):
		refs = []
		gens = []
		for gen_sen, resp_sen in zip(input['res_key'], input['reference_key']):
			gen_sen_processed = dataloader.trim_index(gen_sen)
			resp_sen_processed = dataloader.trim_index(resp_sen[1:])
			refs.append([resp_sen_processed])
			gens.append(gen_sen_processed)
		print('refs:', refs)
		print('gens:', gens)
		return corpus_bleu(refs, gens, smoothing_function=SmoothingFunction().method7)

	def test_close1(self):
		dataloader = FakeDataLoader()
		data = dataloader.get_data('reference_key', 'reference_len_key', 'gen_prob_key',\
								   'gen_key', 'post_key', 'resp_key', 'res_key', \
								   True, False)
		ans = self.get_bleu(dataloader, data)

		bcm = BleuCorpusMetric(dataloader, 'reference_key', 'res_key')
		bcm.forward(data)
		assert ans == bcm.close()['bleu']

	def test_close2(self):
		dataloader = FakeDataLoader()
		data = dataloader.get_data('reference_key', 'reference_len_key', 'gen_prob_key',\
								   'gen_key', 'post_key', 'resp_key', 'res_key', \
								   True, False)
		data['reference_key'] = np.delete(data['reference_key'], 1, axis=0)

		bcm = BleuCorpusMetric(dataloader, 'reference_key', 'res_key')

		with pytest.raises(ValueError):
			bcm.forward(data)

	def test_close3(self):
		dataloader = FakeDataLoader()
		data = dataloader.get_data('reference_key', 'reference_len_key', 'gen_prob_key',\
								   'gen_key', 'post_key', 'resp_key', 'res_key', \
								   True, True)
		ans = self.get_bleu(dataloader, data)

		bcm = BleuCorpusMetric(dataloader, 'reference_key', 'res_key')
		bcm.forward(data)

		assert bcm.close()['bleu'] == ans

	def test_close4(self):
		dataloader = FakeDataLoader()

		ref1 = ['<go>', 'what', 'how', 'here', 'do', 'what', 'here', '<eos>']
		ref2 = ['<go>', 'how', 'here', 'what', 'do', 'how', 'here', '<eos>']
		sen1 = ['how', 'do', 'what', 'here', '<eos>']
		sen2 = ['do', 'what', 'how', 'here', '<eos>']

		sen_to_idx = (lambda x: dataloader.vocab_to_index[x])
		map(sen_to_idx, ref1)
		map(sen_to_idx, ref2)
		map(sen_to_idx, sen1)
		map(sen_to_idx, sen2)
		data = {'reference_key': [ref1, ref2], 'res_key': [sen1, sen2]}

		bcm = BleuCorpusMetric(dataloader, 'reference_key', 'res_key')
		bcm.forward(data)

		ans = self.get_bleu(dataloader, data)

		assert ans > 0
		assert ans == bcm.close()['bleu']

	def test_close5(self):
		dataloader = FakeDataLoader()

		ref1 = ['<go>', 'what', 'how', 'here', 'do', 'what', 'here', '<eos>']
		ref2 = ['<go>', 'how', 'here', 'what', 'do', 'how', 'here', '<eos>']
		sen1 = ['what', 'how', 'here', 'do', 'what', 'here', '<eos>']
		sen2 = ['how', 'here', 'what', 'do', 'how', 'here', '<eos>']

		sen_to_idx = (lambda x: dataloader.vocab_to_index[x])
		ref1 = list(map(sen_to_idx, ref1))
		ref2 = list(map(sen_to_idx, ref2))
		sen1 = list(map(sen_to_idx, sen1))
		sen2 = list(map(sen_to_idx, sen2))
		data = {'reference_key': [ref1, ref2], 'res_key': [sen1, sen2]}

		bcm = BleuCorpusMetric(dataloader, 'reference_key', 'res_key')
		bcm.forward(data)

		ans = self.get_bleu(dataloader, data)

		assert ans > 0
		assert ans == bcm.close()['bleu']

class TestMultiTurnBleuCorpusMetric:
	def get_bleu(self, dataloader, input):
		refs = []
		gens = []
		for i in range(len(input['reference_key'])):
			for resp_sen, gen_sen in zip(input['reference_key'][i], input['res_key'][i]):
				gen_sen_processed = dataloader.trim_index(gen_sen)
				resp_sen_processed = dataloader.trim_index(resp_sen)
				gens.append(gen_sen_processed)
				refs.append([resp_sen_processed[1:]])
		print('refs:', refs)
		print('gens:', gens)
		return corpus_bleu(refs, gens, smoothing_function=SmoothingFunction().method7)

	def test_close1(self):
		dataloader = FakeDataLoader()
		data = dataloader.get_data('reference_key', 'reference_len_key', 'gen_prob_key',\
								   'gen_key', 'post_key', 'resp_key', 'res_key', \
								   True, False, True)

		ans = self.get_bleu(dataloader, data)

		bcm = MultiTurnBleuCorpusMetric(dataloader, 'reference_key', 'res_key')
		bcm.forward(data)

		assert bcm.close()['bleu'] == ans

	def test_close2(self):
		dataloader = FakeDataLoader()
		data = dataloader.get_data('reference_key', 'reference_len_key', 'gen_prob_key',\
								   'gen_key', 'post_key', 'resp_key', 'res_key', \
								   True, False, True)
		data['reference_key'] = np.delete(data['reference_key'], 0, axis=0)
		bcm = MultiTurnBleuCorpusMetric(dataloader, 'reference_key', 'gen_key')

		with pytest.raises(ValueError):
			bcm.forward(data)

	def test_close3(self):
		dataloader = FakeDataLoader()
		data = dataloader.get_data('reference_key', 'reference_len_key', 'gen_prob_key',\
								   'gen_key', 'post_key', 'resp_key', 'res_key', \
								   True, True, True)
		ans = self.get_bleu(dataloader, data)

		bcm = MultiTurnBleuCorpusMetric(dataloader, 'reference_key', 'res_key')
		bcm.forward(data)

		assert bcm.close()['bleu'] == ans

	def test_close4(self):
		dataloader = FakeDataLoader()

		ref1 = ['<go>', 'what', 'how', 'here', 'do', 'what', 'here', '<eos>']
		ref2 = ['<go>', 'how', 'here', 'what', 'do', 'how', 'here', '<eos>']
		ref3 = ['<go>', 'here', 'here', 'here', 'here', 'here', 'here', '<eos>']
		sen1 = ['what', 'how', 'here', 'do', 'what', 'here', '<eos>']
		sen2 = ['how', 'here', 'what', 'do', 'how', 'here', '<eos>']
		sen3 = ['here', 'here', 'here', 'here', 'here', 'here', '<eos>']

		sen_to_idx = (lambda x: dataloader.vocab_to_index[x])
		ref1 = list(map(sen_to_idx, ref1))
		ref2 = list(map(sen_to_idx, ref2))
		ref3 = list(map(sen_to_idx, ref3))
		sen1 = list(map(sen_to_idx, sen1))
		sen2 = list(map(sen_to_idx, sen2))
		sen3 = list(map(sen_to_idx, sen3))
		print(ref1)
		data = {'reference_key': [[ref1, ref2], [ref3]], 'res_key': [[sen1, sen2], [sen3]]}

		bcm = MultiTurnBleuCorpusMetric(dataloader, 'reference_key', 'res_key')
		bcm.forward(data)

		ans = self.get_bleu(dataloader, data)

		assert ans > 0
		assert ans == bcm.close()['bleu']

class TestLanguageGenerationRecorder():
	def test_init(self):
		sdr = LanguageGenerationRecorder(None, 'a')
		assert sdr.gen_key == 'a'

	def test_close1(self):
		vocab_list = ['<pad>', '<unk>', '<go>', '<eos>', 'what', 'how', 'here', 'do']
		data = {'gen': [], 'res': []}
		batch = 5
		length = 10
		for i in range(batch):
			sen = []
			res = []
			for j in range(length):
				t = randrange(4, 8)
				sen.append(t)
				res.append(vocab_list[t])
			sen[-1] = 3
			res[-1] = '<eos>'
			while res[-1] == '<eos>':
				res.pop()
			data['gen'].append(sen)
			data['res'].append(res)

		data['gen'] = np.array(data['gen'])

		sdr = LanguageGenerationRecorder(FakeDataLoader(), 'gen')
		sdr.forward(data)
		assert sdr.close()['gen'] == data['res']

	def test_close2(self):
		vocab_list = ['<pad>', '<unk>', '<go>', '<eos>', 'what', 'how', 'here', 'do']
		data = {'gen': [], 'res': []}
		batch = 5
		length = 10
		for i in range(batch):
			sen = []
			res = []
			for j in range(length):
				t = randrange(4, 8)
				sen.append(t)
				res.append(vocab_list[t])
			sen[-1] = 3
			res[-1] = '<eos>'
			while res[-1] == '<eos>':
				res.pop()
			data['gen'].append(sen)
			data['res'].append(res)

		sdr = LanguageGenerationRecorder(FakeDataLoader(), 'gen')
		sdr.forward(data)
		assert sdr.close()['gen'] == data['res']

class TestSingleTurnDialogRecorder():
	def test_close1(self):
		vocab_list = ['<pad>', '<unk>', '<go>', '<eos>', 'what', 'how', 'here', 'do']
		data = {'post': [], 'resp': [], 'gen': []}
		resdata = {'post': [], 'resp': [], 'gen': []}
		batch = 5
		length = 10
		for i in range(batch):
			for key in data.keys():
				sen = []
				res = []
				for j in range(length):
					t = randrange(4, 8)
					sen.append(t)
					res.append(vocab_list[t])
				if not key == 'gen':
					res.remove(res[0])
				data[key].append(sen)
				resdata[key].append(res)

		for key in data.keys():
			data[key] = np.array(data[key])
		sdr = SingleTurnDialogRecorder(FakeDataLoader(), 'post', 'resp', 'gen')
		sdr.forward(data)
		assert sdr.close() == resdata

	def test_close2(self):
		vocab_list = ['<pad>', '<unk>', '<go>', '<eos>', 'what', 'how', 'here', 'do']
		data = {'post': [], 'resp': [], 'gen': []}
		resdata = {'post': [], 'resp': [], 'gen': []}
		batch = 5
		length = 10
		for i in range(batch):
			for key in data.keys():
				sen = []
				res = []
				for j in range(length):
					t = randrange(4, 8)
					sen.append(t)
					res.append(vocab_list[t])
				if not key == 'gen':
					res.remove(res[0])
				data[key].append(sen)
				resdata[key].append(res)

		sdr = SingleTurnDialogRecorder(FakeDataLoader(), 'post', 'resp', 'gen')
		sdr.forward(data)
		assert sdr.close() == resdata

class TestMultiTurnDialogRecorder:
	def multi_turn_inx_to_sen(self, sen_lists):
		dataloader = FakeDataLoader()
		res_list = []
		if not isinstance(sen_lists, np.ndarray):
			sen_lists = np.array(sen_lists)
		for i, _ in enumerate(sen_lists):
			if _[0][0] == dataloader.vocab_to_index['<go>']:
				res_list.append(dataloader.multi_turn_index_to_sen(_[:, 1:]))
			else:
				res_list.append(dataloader.multi_turn_index_to_sen(_[:, :]))
		return res_list

	def test_close1(self):
		dataloader = FakeDataLoader()
		dr = MultiTurnDialogRecorder(dataloader, 'post_key', 'reference_key', 'res_key')

		data = dataloader.get_data('reference_key', 'reference_len_key', 'gen_prob_key',\
								   'gen_key', 'post_key', 'resp_key', 'res_key', \
								   True, False, True)

		res = {}
		res['context'] = self.multi_turn_inx_to_sen(data['post_key'])
		res['reference'] = self.multi_turn_inx_to_sen(data['reference_key'])
		res['gen'] = self.multi_turn_inx_to_sen(data['res_key'])
		dr.forward(data)
		assert dr.close() == res

	def test_close2(self):
		dataloader = FakeDataLoader()
		dr = MultiTurnDialogRecorder(dataloader, 'post_key', 'reference_key', 'res_key')

		data = dataloader.get_data('reference_key', 'reference_len_key', 'gen_prob_key',\
								   'gen_key', 'post_key', 'resp_key', 'res_key', \
								   True, True, True)

		res = {}
		res['context'] = self.multi_turn_inx_to_sen(data['post_key'])
		res['reference'] = self.multi_turn_inx_to_sen(data['reference_key'])
		res['gen'] = self.multi_turn_inx_to_sen(data['res_key'])
		dr.forward(data)
		assert dr.close() == res

class TestMetricChain():
	def test_init(self):
		mc = MetricChain()

	def test_add_metric(self):
		mc = MetricChain()
		with pytest.raises(TypeError):
			mc.add_metric([1, 2, 3])

	def test_close1(self):
		dataloader = FakeDataLoader()
		data = dataloader.get_data('reference_key', 'reference_len_key', 'gen_prob_key',\
								   'gen_key', 'post_key', 'resp_key', 'res_key', \
								   True, False, True)
		pm = MultiTurnPerplexityMetric(dataloader, 'reference_key', 'reference_len_key', 'gen_prob_key', full_check=True)
		perplexity = TestMultiTurnPerplexityMetric().get_perplexity(data)

		bcm = MultiTurnBleuCorpusMetric(dataloader, 'reference_key', 'res_key')
		bleu = TestMultiTurnBleuCorpusMetric().get_bleu(dataloader, data)

		mc = MetricChain()
		mc.add_metric(pm)
		mc.add_metric(bcm)
		mc.forward(data)
		res = mc.close()

		assert np.isclose(res['perplexity'], perplexity)
		assert np.isclose(res['bleu'], bleu)
