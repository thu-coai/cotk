import copy
import random

import numpy as np
import pytest

from cotk.metric import MultiTurnPerplexityMetric, MultiTurnBleuCorpusMetric, MetricChain

from test_perplexity import TestMultiTurnPerplexityMetric
from test_bleu import TestMultiTurnBleuCorpusMetric
from metric_base import *

def setup_module():
	random.seed(0)
	np.random.seed(0)

pytestmark = pytest.mark.skip("all tests still WIP")

class TestMetricChain():
	def test_init(self):
		mc = MetricChain()

	def test_add_metric(self):
		mc = MetricChain()
		with pytest.raises(TypeError):
			mc.add_metric([1, 2, 3])

	def test_close1(self):
		dataloader = FakeMultiDataloader()
		data = dataloader.get_data(reference_key='reference_key', reference_len_key='reference_len_key', \
								   turn_len_key='turn_len_key', gen_prob_key='gen_prob_key', \
								   gen_key='gen_key', context_key='context_key')
		pm = MultiTurnPerplexityMetric(dataloader, 'reference_key', 'reference_len_key', 'gen_prob_key', \
									   invalid_vocab=True, full_check=True)
		perplexity = TestMultiTurnPerplexityMetric().get_perplexity( \
			data, dataloader, True, 'reference_key', 'reference_len_key', 'gen_prob_key')

		bcm = MultiTurnBleuCorpusMetric(dataloader, multi_turn_reference_allvocabs_key='reference_key', \
										multi_turn_gen_key='gen_key', turn_len_key='turn_len_key')
		bleu = TestMultiTurnBleuCorpusMetric().get_bleu(dataloader, data, 'reference_key', 'gen_key')

		_data = copy.deepcopy(data)
		mc = MetricChain()
		mc.add_metric(pm)
		mc.add_metric(bcm)
		mc.forward(data)
		res = mc.close()

		assert np.isclose(res['perplexity'], perplexity)
		assert np.isclose(res['bleu'], bleu)
		assert same_dict(data, _data)
