import pytest
import numpy as np
from cotk.dataloader import SentenceClassification, SST, Sentence
from cotk.metric import MetricBase

from test_dataloader import BaseTestLanguageProcessing
from version_test_base import base_test_version


def setup_module():
	import random
	random.seed(0)
	import numpy as np
	np.random.seed(0)

class TestSentenceClassification(BaseTestLanguageProcessing):
	def base_test_init(self, dl: SentenceClassification):
		super().base_test_init(dl)
		assert isinstance(dl, SentenceClassification)
		assert list(dl.get_special_tokens_mapping().values())[:4] == ["<pad>", "<unk>", "<go>", "<eos>"]
		assert [dl.pad_id, dl.unk_id, dl.go_id, dl.eos_id] == [0, 1, 2, 3]
		assert isinstance(dl.all_vocab_list, list)
		assert dl.all_vocab_list[:len(dl.get_special_tokens_mapping())] == list(dl.get_special_tokens_mapping().values())
		assert dl.all_vocab_size == len(dl.all_vocab_list)
		for i, word in enumerate(dl.all_vocab_list):
			assert isinstance(word, str)
			assert dl.convert_tokens_to_ids([word])[0] == i
		assert dl.all_vocab_size == len(dl.all_vocab_list)

		# assert the data has valid token
		assert dl.frequent_vocab_size > 4
		# assert the data has invalid token
		assert dl.all_vocab_size > dl.frequent_vocab_size

	def base_test_all_unknown(self, dl: SentenceClassification):
		# if min_rare_vocab_times very big, there is no rare words.
		assert dl.frequent_vocab_size == dl.all_vocab_size

	def base_test_metric(self, dl):
		assert isinstance(dl.get_metric(), MetricBase)

	def base_test_multi_runs(self, dl_list):
		assert all(x.all_vocab_list == dl_list[0].all_vocab_list for x in dl_list)

	def _test_get_batch_of_sentence_field(self, batch_data: dict, field_name, indexes:list):
		"""Help function for testing :meth:`Dataloader.get_batch.` The dataloader must have a sentence field.

		Args:
			batch_data: return value of get_batch.
			field_name: argument of Sentence.get_batch
			indexes: argument of Sentence.get_batch
		"""
		assert isinstance(batch_data.get(field_name, None), np.ndarray)
		assert len(batch_data[field_name].shape) == 2 and batch_data[field_name].shape[0] == len(indexes)
		assert isinstance(batch_data.get(field_name + '_length', None), np.ndarray)
		assert batch_data[field_name + '_length'].shape == (len(indexes), )
		assert isinstance(batch_data.get(field_name + '_allvocabs', None), np.ndarray)
		assert batch_data[field_name + '_allvocabs'].shape[0] == len(indexes) and len(batch_data[field_name + '_allvocabs'].shape) == 2
		assert isinstance(batch_data.get(field_name + '_str', None), list)
		assert isinstance(batch_data[field_name + '_str'][0], str)

		shape = batch_data[field_name].shape
		assert shape[1] == max(batch_data[field_name + '_length'])

	def base_test_get_batch(self, dl: SentenceClassification):
		super().base_test_get_batch(dl)
		for set_name in dl.fields:
			dl.restart(set_name, shuffle=True, batch_size=1)
			size = len(next(iter(dl.data[set_name].values())))
			for i in range(size):
				indexes = [i]
				batch_data = dl.get_batch(set_name, indexes)
				for field_name in dl.fields[set_name]:
					if isinstance(dl.fields[set_name][field_name], Sentence):
						self._test_get_batch_of_sentence_field(batch_data, field_name, indexes)


@pytest.fixture
def load_sst():
	def _load_sst(min_rare_vocab_times=0):
		return SST("./tests/dataloader/dummy_sst#SST", min_rare_vocab_times=min_rare_vocab_times)
	return _load_sst

class TestSST(TestSentenceClassification):
	def test_version(self):
		base_test_version(SST)

	@pytest.mark.dependency()
	def test_init(self, load_sst):
		super().base_test_init(load_sst())
		super().base_test_all_unknown(load_sst(1000000))

	def test_restart(self, load_sst):
		super().base_test_restart(load_sst())

	@pytest.mark.dependency(depends=["TestSST::test_init"])
	def test_get_batch(self, load_sst):
		super().base_test_get_batch(load_sst())

	@pytest.mark.dependency(depends=["TestSST::test_init"])
	def test_get_next_batch(self, load_sst):
		super().base_test_get_next_batch(load_sst())

	@pytest.mark.dependency(depends=["TestSST::test_init"])
	def test_convert(self, load_sst):
		super().base_test_convert(load_sst())

	def test_metric(self, load_sst):
		super().base_test_metric(load_sst())

	def test_init_multi_runs(self, load_sst):
		super().base_test_multi_runs([load_sst() for i in range(3)])
