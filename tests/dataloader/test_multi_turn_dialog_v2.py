import random
from itertools import chain
from collections import OrderedDict

import pytest
import numpy as np

from cotk.dataloader import MultiTurnDialog, Session, SwitchboardCorpus, UbuntuCorpus
from cotk.metric import MetricBase
from cotk.wordvector import Glove
from test_dataloader import BaseTestLanguageProcessing
from version_test_base import base_test_version


def setup_module():
	random.seed(0)
	np.random.seed(0)


class TestMultiTurnDialog(BaseTestLanguageProcessing):
	def base_test_init(self, dl: MultiTurnDialog):
		assert isinstance(dl, MultiTurnDialog)
		super().base_test_init(dl)
		assert dl.default_field_name is not None and dl.default_field_set_name is not None
		default_field = dl.get_default_field()
		assert isinstance(default_field, Session)

		assert isinstance(dl.frequent_vocab_list, list)
		assert dl.frequent_vocab_size == len(dl.frequent_vocab_list)
		assert isinstance(dl.all_vocab_list, list)
		assert dl.all_vocab_size == len(dl.all_vocab_list)
		assert dl.all_vocab_size > 4
		assert dl.all_vocab_size >= dl.frequent_vocab_size

	def _test_get_batch_of_session_field(self, batch_data: dict, field_name: str, indexes: list):
		"""Help function for testing :meth:`Dataloader.get_batch.` The dataloader must have a session field.

		Args:
			batch_data: return value of get_batch.
			field_name: argument of Session.get_batch
			indexes: argument of Session.get_batch
		"""
		assert isinstance(batch_data.get(field_name, None), np.ndarray)
		assert len(batch_data[field_name].shape) == 3 and batch_data[field_name].shape[0] == len(indexes)
		assert isinstance(batch_data.get(field_name + '_turn_length'), np.ndarray)
		assert len(batch_data[field_name + '_turn_length'].shape) == 1 and \
			   batch_data[field_name + '_turn_length'].shape[0] == len(indexes)
		assert isinstance(batch_data.get(field_name + '_sent_length', None), list)
		assert len(batch_data[field_name + '_sent_length']) == len(indexes) and isinstance(
			batch_data[field_name + '_sent_length'][0], list) and not isinstance(batch_data[field_name+'_sent_length'][0][0], list)
		assert isinstance(batch_data.get(field_name + '_allvocabs', None), np.ndarray)
		assert len(batch_data[field_name + '_allvocabs'].shape) == 3 and batch_data[field_name + '_allvocabs'].shape[
			0] == len(indexes)
		assert isinstance(batch_data.get(field_name + '_str', None), list)
		assert isinstance(batch_data[field_name + '_str'][0], list)
		assert isinstance(batch_data[field_name + '_str'][0][0], str)

		shape = batch_data[field_name].shape
		assert max(batch_data[field_name + '_turn_length']) == shape[1]
		assert max(chain.from_iterable(batch_data[field_name + '_sent_length'])) == shape[2]

	def base_test_get_batch(self, dl):
		super().base_test_get_batch(dl)
		for set_name in dl.fields:
			dl.restart(set_name, shuffle=True, batch_size=1)
			size = len(next(iter(dl.data[set_name].values())))
			for i in range(size):
				indexes = [i]
				batch_data = dl.get_batch(set_name, indexes)
				for field_name in dl.fields[set_name]:
					if isinstance(dl.fields[set_name][field_name], Session):
						self._test_get_batch_of_session_field(batch_data, field_name, indexes)

	def base_test_multi_turn_convert(self, dl: MultiTurnDialog):
		sent_id = [[0, 1, 2], [2, 1, 1]]
		sent = [["<pad>", "<unk>", "<go>"], ["<go>", "<unk>", "<unk>"]]
		assert sent == dl.convert_multi_turn_ids_to_tokens(sent_id, remove_special=False)
		assert sent_id == dl.convert_multi_turn_tokens_to_ids(sent)

		sent = [["<unk>", "<go>", "<pad>", "<unkownword>", "<pad>", "<go>"], ["<go>", "<eos>"]]
		sent_id = [[1, 2, 0, 1, 0, 2], [2, 3]]
		assert sent_id == dl.convert_multi_turn_tokens_to_ids(sent)

		sent_id = [[0, 1, 2, 2, 0, 3, 1, 0, 0], [0, 3, 2], [1, 2, 2, 0], [1, 2, 2, 3]]
		sent = [["<pad>", "<unk>", "<go>", "<go>", "<pad>", "<eos>", "<unk>", "<pad>", "<pad>"], \
				["<pad>", "<eos>", "<go>"], \
				["<unk>", "<go>", "<go>", "<pad>"], \
				["<unk>", "<go>", "<go>", "<eos>"]]
		assert sent == dl.convert_multi_turn_ids_to_tokens(sent_id, remove_special=False, trim=False)
		sent = [["<pad>", "<unk>", "<go>", "<go>"]]
		sent_id = [[0, 1, 2, 2]]
		assert sent == dl.convert_multi_turn_ids_to_tokens(sent_id)

		sent = [[dl.all_vocab_list[dl.unk_id]]]
		assert [[dl.unk_id]] == dl.convert_multi_turn_tokens_to_ids(sent)
		assert [[dl.unk_id]] == dl.convert_multi_turn_tokens_to_ids(sent, only_frequent_word=True)

	def base_test_teacher_forcing_metric(self, dl):
		assert isinstance(dl.get_teacher_forcing_metric(), MetricBase)

	def base_test_teacher_inference_metric(self, dl):
		assert isinstance(dl.get_inference_metric(), MetricBase)

	def base_test_multi_runs(self, dl_list):
		assert all(x.all_vocab_list == dl_list[0].all_vocab_list for x in dl_list)

@pytest.fixture
def load_ubuntucorpus():
	def _load_ubuntucorpus(min_rare_vocab_times=0):
		return UbuntuCorpus("./tests/dataloader/dummy_ubuntucorpus#Ubuntu", min_rare_vocab_times=min_rare_vocab_times)
	return _load_ubuntucorpus

class TestUbuntuCorpus(TestMultiTurnDialog):
	# def test_version(self):
	# 	base_test_version(UbuntuCorpus)

	@pytest.mark.dependency()
	def test_init(self, load_ubuntucorpus):
		dl = load_ubuntucorpus()
		super().base_test_init(dl)
		assert isinstance(dl, UbuntuCorpus)
		assert set(dl.fields.keys()) == set(dl.data.keys()) == {'train', 'test', 'dev'}
		for fields_of_one_set in dl.fields.values():
			assert isinstance(fields_of_one_set, OrderedDict)
			assert len(fields_of_one_set) == 1
			assert isinstance(fields_of_one_set.get('session', None), Session)


	def test_restart(self, load_ubuntucorpus):
		super().base_test_restart(load_ubuntucorpus())

	@pytest.mark.dependency(depends=["TestUbuntuCorpus::test_init"])
	def test_get_batch(self, load_ubuntucorpus):
		super().base_test_get_batch(load_ubuntucorpus())

	@pytest.mark.dependency(depends=["TestUbuntuCorpus::test_init"])
	def test_get_next_batch(self, load_ubuntucorpus):
		super().base_test_get_next_batch(load_ubuntucorpus())

	@pytest.mark.dependency(depends=["TestUbuntuCorpus::test_init"])
	def test_convert(self, load_ubuntucorpus):
		super().base_test_convert(load_ubuntucorpus())

	def test_multi_turn_convert(self, load_ubuntucorpus):
		super().base_test_multi_turn_convert(load_ubuntucorpus())

	def test_teacher_forcing_metric(self, load_ubuntucorpus):
		super().base_test_teacher_forcing_metric(load_ubuntucorpus())

	def test_teacher_inference_metric(self, load_ubuntucorpus):
		super().base_test_teacher_inference_metric(load_ubuntucorpus())

	def test_init_multi_runs(self, load_ubuntucorpus):
		super().base_test_multi_runs([load_ubuntucorpus() for i in range(3)])


@pytest.fixture
def load_switchboardcorpus():
	def _load_switchboardcorpus(min_rare_vocab_times=0):
		return SwitchboardCorpus("./tests/dataloader/dummy_switchboardcorpus#SwitchboardCorpus",
								 min_rare_vocab_times=min_rare_vocab_times)

	return _load_switchboardcorpus


class TestSwitchboardCorpus(TestMultiTurnDialog):
	def test_version(self):
		base_test_version(SwitchboardCorpus)

	SwitchboardCorpusSetNames = ('train', 'test', 'dev', 'multi_ref')

	@pytest.mark.dependency()
	def test_init(self, load_switchboardcorpus):
		dl = load_switchboardcorpus()
		super().base_test_init(dl)
		assert isinstance(dl, SwitchboardCorpus)
		set_names = self.SwitchboardCorpusSetNames
		assert set(dl.fields.keys()) == set(set_names)
		for set_name in set_names[:-1]:
			assert isinstance(dl.fields[set_name], OrderedDict)
			assert len(dl.fields[set_name]) == 1
			assert isinstance(dl.fields[set_name].get('session', None), Session)
		assert isinstance(dl.fields['multi_ref'], OrderedDict)
		assert len(dl.fields['multi_ref']) == 2
		assert all(a == b for a, b in zip(dl.fields['multi_ref'].keys(), ['session', 'candidate']))
		f1, f2 = dl.fields['multi_ref'].values()
		assert isinstance(f1, Session) and isinstance(f2, SwitchboardCorpus.Candidate)

	@pytest.mark.dependency()
	def test_restart(self, load_switchboardcorpus):
		super().base_test_restart(load_switchboardcorpus())

	@pytest.mark.dependency()
	def test_get_batch(self, load_switchboardcorpus):
		dl: SwitchboardCorpus = load_switchboardcorpus()
		super().base_test_get_batch(dl)

	@pytest.mark.dependency()
	def test_get_next_batch(self, load_switchboardcorpus):
		super().base_test_get_next_batch(load_switchboardcorpus())

	@pytest.mark.dependency()
	def test_convert(self, load_switchboardcorpus):
		super().base_test_convert(load_switchboardcorpus())

	@pytest.mark.dependency()
	def test_teacher_forcing_metric(self, load_switchboardcorpus):
		super().base_test_teacher_forcing_metric(load_switchboardcorpus())

	@pytest.mark.dependency()
	def test_teacher_inference_metric(self, load_switchboardcorpus):
		super().base_test_teacher_inference_metric(load_switchboardcorpus())

	# TODO: fix bug
	@pytest.mark.skip()
	@pytest.mark.dependency()
	def test_teacher_precision_recall_metric(self, load_switchboardcorpus):
		dl = load_switchboardcorpus()
		glove = Glove("./tests/wordvector/dummy_glove/300d/")
		embed = glove.load_dict(dl.all_vocab_list)
		assert isinstance(dl.get_multi_ref_metric(generated_num_per_context=3, word2vec=embed), MetricBase)

	@pytest.mark.dependency()
	def test_init_multi_runs(self, load_switchboardcorpus):
		super().base_test_multi_runs([load_switchboardcorpus() for i in range(3)])

	def test_multi_turn_convert(self, load_switchboardcorpus):
		super().base_test_multi_turn_convert(load_switchboardcorpus())