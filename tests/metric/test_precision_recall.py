import copy
import random
import numpy as np
import pytest

from cotk.metric import BleuPrecisionRecallMetric, EmbSimilarityPrecisionRecallMetric

from metric_base import *

def setup_module():
	random.seed(0)
	np.random.seed(0)

pytestmark = pytest.mark.skip("all tests still WIP")

bleu_precision_recall_test_parameter = generate_testcase(\
	(zip(test_argument), "add"),
	(zip(test_shape, test_type), "multi"),
	(zip(test_batch_len), "add"),
	(zip(test_ref_len), "multi"),
	(zip(test_gen_len), "multi"),
	(zip(test_ngram), "add")
)

class TestBleuPrecisionRecallMetric():
	default_reference_key = 'candidate_allvocabs'
	default_gen_key = 'multiple_gen'
	default_keywords = (default_reference_key, default_gen_key)

	def test_base_class(self):
		with pytest.raises(NotImplementedError):
			dataloader = FakeMultiDataloader()
			gen = []
			reference = []
			bprm = BleuPrecisionRecallMetric(dataloader, 1, 3)
			super(BleuPrecisionRecallMetric, bprm)._score(gen, reference)

	def test_hashvalue(self):
		dataloader = FakeMultiDataloader()
		reference_key, gen_key = self.default_keywords
		data = dataloader.get_data(reference_key=reference_key, gen_key=gen_key, \
								   to_list=True, pad=False, \
								   ref_len='non-empty', gen_len='non-empty', test_prec_rec=True)
		bprm = BleuPrecisionRecallMetric(dataloader, 4, 3)
		assert bprm.candidate_allvocabs_key == reference_key
		bprm_shuffle = BleuPrecisionRecallMetric(dataloader, 4, 3)

		data_shuffle = shuffle_instances(data, self.default_keywords)
		for idx in range(len(data_shuffle[reference_key])):
			np.random.shuffle(data_shuffle[reference_key][idx])
		batches_shuffle = split_batch(data_shuffle, self.default_keywords)

		bprm.forward(data)
		res = bprm.close()

		for batch in batches_shuffle:
			bprm_shuffle.forward(batch)
		res_shuffle = bprm_shuffle.close()
		assert same_dict(res, res_shuffle, False)

		data_less_word = copy.deepcopy(data)
		data_less_word[reference_key][0][0] = data_less_word[reference_key][0][0][:-2]
		for data_unequal in [data_less_word] + generate_unequal_data(data, self.default_keywords, \
												  dataloader.pad_id, \
												  reference_key, reference_is_3D=True):
			bprm_unequal = BleuPrecisionRecallMetric(dataloader, 4, 3)

			bprm_unequal.forward(data_unequal)
			res_unequal = bprm_unequal.close()

			assert res['BLEU-4 hashvalue'] != res_unequal['BLEU-4 hashvalue']

	@pytest.mark.parametrize('argument, shape, type, batch_len, ref_len, gen_len, ngram', \
		bleu_precision_recall_test_parameter)
	def test_close(self, argument, shape, type, batch_len, ref_len, gen_len, ngram):
		dataloader = FakeMultiDataloader()

		if ngram not in range(1, 5):
			with pytest.raises(ValueError, match=r"ngram should belong to \[1, 4\]"):
				bprm = BleuPrecisionRecallMetric(dataloader, ngram, 3)
			return

		if argument == 'default':
			reference_key, gen_key = self.default_keywords
			bprm = BleuPrecisionRecallMetric(dataloader, ngram, 3)
		else:
			reference_key, gen_key = ('rk', 'gk')
			bprm = BleuPrecisionRecallMetric(dataloader, ngram, 3, reference_key, gen_key)

		# TODO: might need adaptation of dataloader.get_data for test_prec_rec
		# turn_length is not generated_num_per_context conceptually
		data = dataloader.get_data(reference_key=reference_key, gen_key=gen_key, \
								   to_list=(type == 'list'), pad=(shape == 'pad'), \
								   ref_len=ref_len, gen_len=gen_len, test_prec_rec=True)
		_data = copy.deepcopy(data)
		if batch_len == 'unequal':
			data[reference_key] = data[reference_key][1:]
			_data = copy.deepcopy(data)
			with pytest.raises(ValueError, match="Batch num is not matched."):
				bprm.forward(data)
		else:
			bprm.forward(data)
			ans = bprm.close()
			prefix = 'BLEU-' + str(ngram)
			assert sorted(ans.keys()) == [prefix + ' hashvalue', prefix + ' precision', prefix + ' recall']

		assert same_dict(data, _data)

	def test_version(self):
		version_test(BleuPrecisionRecallMetric, dataloader=FakeMultiDataloader())


emb_similarity_precision_recall_test_parameter = generate_testcase( \
	(zip(test_argument), "add"),
	(zip(test_shape, test_type), "multi"),
	(zip(test_batch_len), "add"),
	(zip(test_ref_len), "add"),
	(zip(test_gen_len), "add"),
	(zip(test_ref_vocab), "multi"),
	(zip(test_gen_vocab), "multi"),
	(zip(test_emb_mode), "add"),
	(zip(test_emb_type), "add"),
	(zip(test_emb_len), "add")
)


class TestEmbSimilarityPrecisionRecallMetric():
	default_reference_key = 'candidate_allvocabs'
	default_gen_key = 'multiple_gen'
	default_keywords = (default_reference_key, default_gen_key)

	def test_hashvalue(self):
		dataloader = FakeMultiDataloader()
		emb = {}
		emb_unequal = {}
		for word in dataloader.all_vocab_list[:dataloader.valid_vocab_len]:
			vec = []
			for j in range(5):
				vec.append(random.random())
			vec = np.array(vec)
			emb[word] = vec
			emb_unequal[word] = vec + 1

		reference_key, gen_key = self.default_keywords
		key_list = [reference_key, gen_key]
		data = dataloader.get_data(reference_key=reference_key, gen_key=gen_key, \
								   to_list=True, pad=False, \
								   ref_len='non-empty', gen_len='non-empty', \
								   ref_vocab='valid_vocab', gen_vocab='valid_vocab', test_prec_rec=True)
		espr = EmbSimilarityPrecisionRecallMetric(dataloader, emb, 'avg', 3)
		espr_shuffle = EmbSimilarityPrecisionRecallMetric(dataloader, emb, 'avg', 3)

		data_shuffle = shuffle_instances(data, key_list)
		for idx in range(len(data_shuffle[reference_key])):
			np.random.shuffle(data_shuffle[reference_key][idx])
		batches_shuffle = split_batch(data_shuffle, key_list)

		espr.forward(data)
		res = espr.close()

		for batch in batches_shuffle:
			espr_shuffle.forward(batch)
		res_shuffle = espr_shuffle.close()

		assert same_dict(res, res_shuffle, False)

		data_less_word = copy.deepcopy(data)
		data_less_word[reference_key][0][0] = data_less_word[reference_key][0][0][:-2]
		for data_unequal in [data_less_word] + generate_unequal_data(data, key_list, \
												dataloader.pad_id, \
												reference_key, reference_is_3D=True):
			espr_unequal = EmbSimilarityPrecisionRecallMetric(dataloader, emb, 'avg', 3)

			espr_unequal.forward(data_unequal)
			res_unequal = espr_unequal.close()

			assert res['avg-bow hashvalue'] != res_unequal['avg-bow hashvalue']
		espr_unequal = EmbSimilarityPrecisionRecallMetric(dataloader, emb_unequal, 'avg', 3)
		espr_unequal.forward(data)
		res_unequal = espr_unequal.close()
		assert res['avg-bow hashvalue'] != res_unequal['avg-bow hashvalue']

	@pytest.mark.parametrize('argument, shape, type, batch_len, ref_len, gen_len, ' \
							 'ref_vocab, gen_vocab, emb_mode, emb_type, emb_len', \
							 emb_similarity_precision_recall_test_parameter)
	def test_close(self, argument, shape, type, batch_len, ref_len, gen_len, \
							 ref_vocab, gen_vocab, emb_mode, emb_type, emb_len):
		dataloader = FakeMultiDataloader()

		emb = {}
		for word in dataloader.all_vocab_list[:dataloader.valid_vocab_len]:
			vec = []
			for j in range(5):
				vec.append(random.random())
			emb[word] = vec
		if emb_len == 'unequal':
			key = list(emb.keys())[0]
			emb[key] = emb[key][:-1]
		if emb_type == 'list':
			emb = np.array(list(emb.values()), dtype=np.float32).tolist()

		if emb_type != 'dict':
			with pytest.raises(ValueError, match="invalid type"):
				espr = EmbSimilarityPrecisionRecallMetric(dataloader, emb, emb_mode, 3)
			return
		else:
			if emb_len == 'unequal':
				with pytest.raises(ValueError, match="word embeddings have inconsistent embedding size or are empty"):
					espr = EmbSimilarityPrecisionRecallMetric(dataloader, emb, emb_mode, 3)
				return
		if emb_mode not in ['avg', 'extrema']:
			with pytest.raises(ValueError, match="mode should be 'avg' or 'extrema'."):
				espr = EmbSimilarityPrecisionRecallMetric(dataloader, emb, emb_mode, 3)
			return

		if argument == 'default':
			reference_key, gen_key = self.default_keywords
			print(emb)
			espr = EmbSimilarityPrecisionRecallMetric(dataloader, emb, emb_mode, 3)
		else:
			reference_key, gen_key = ('rk', 'gk')
			espr = EmbSimilarityPrecisionRecallMetric(dataloader, emb, emb_mode, 3, \
													  reference_key, gen_key)

		# TODO: might need adaptation of dataloader.get_data for test_prec_rec
		# turn_length is not generated_num_per_context conceptually
		data = dataloader.get_data(reference_key=reference_key, gen_key=gen_key, \
								   to_list=(type == 'list'), pad=(shape == 'pad'), \
								   ref_len=ref_len, gen_len=gen_len, \
								   ref_vocab=ref_vocab, gen_vocab=gen_vocab, test_prec_rec=True)

		_data = copy.deepcopy(data)
		if batch_len == 'unequal':
			data[reference_key] = data[reference_key][1:]
			_data = copy.deepcopy(data)
			with pytest.raises(ValueError, match="Batch num is not matched."):
				espr.forward(data)
		else:
			# if emb_len < dataloader.all_vocab_size and \
			#	 (ref_vocab == 'all_vocab' or gen_vocab == 'all_vocab'):
			#	 with pytest.raises(ValueError, match="[a-z]* index out of range."):
			#		 espr.forward(data)
			# else:
			espr.forward(data)
			ans = espr.close()
			prefix = emb_mode + '-bow'
			assert sorted(ans.keys()) == [prefix + ' hashvalue', prefix + ' precision', prefix + ' recall']

		assert same_dict(data, _data)

	def test_version(self):
		version_test(EmbSimilarityPrecisionRecallMetric, dataloader=FakeMultiDataloader())
