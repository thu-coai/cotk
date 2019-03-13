import copy
from itertools import chain
import pytest

from cotk.dataloader import MultiTurnDialog, UbuntuCorpus, SwitchboardCorpus
from cotk.metric import MetricBase, HashValueRecorder

def setup_module():
	import random
	random.seed(0)
	import numpy as np
	np.random.seed(0)

class TestMultiTurnDialog():
	def base_test_init(self, dl):
		assert isinstance(dl, MultiTurnDialog)
		assert isinstance(dl.ext_vocab, list)
		assert dl.ext_vocab[:5] == ["<pad>", "<unk>", "<go>", "<eos>"]
		assert [dl.pad_id, dl.unk_id, dl.go_id, dl.eos_id] == [0, 1, 2, 3]
		assert isinstance(dl.key_name, list)
		assert dl.key_name
		for word in dl.key_name:
			assert isinstance(word, str)
		assert isinstance(dl.all_vocab_list, list)
		assert dl.vocab_list[:len(dl.ext_vocab)] == dl.ext_vocab
		assert isinstance(dl.word2id, dict)
		assert len(dl.word2id) == len(dl.all_vocab_list)
		assert dl.vocab_size == len(dl.vocab_list)
		for i, word in enumerate(dl.all_vocab_list):
			assert isinstance(word, str)
			assert dl.word2id[word] == i
		assert dl.all_vocab_size == len(dl.all_vocab_list)
		assert isinstance(dl.data, dict)
		assert len(dl.data) >= 3
		for key in dl.key_name:
			assert isinstance(dl.data[key], dict)
			assert isinstance(dl.data[key]['session'], list)
			assert isinstance(dl.data[key]['session'][0], list)
			content = dl.data[key]['session'][0][0]
			assert content[0] == dl.go_id
			assert content[-1] == dl.eos_id

		# assert the data has valid token
		assert dl.vocab_size > 5
		# assert the data has invalid token
		assert dl.all_vocab_size > dl.vocab_size

	def base_test_all_unknown(self, dl):
		# if invalid_vocab_times very big, there is no invalid words.
		assert dl.vocab_size == dl.vocab_size

	def base_test_restart(self, dl):
		with pytest.raises(ValueError):
			dl.restart("unknown set")
		for key in dl.key_name:
			with pytest.raises(ValueError):
				dl.restart(key)
			record_index = copy.copy(dl.index[key])
			dl.restart(key, batch_size=3, shuffle=False)
			assert record_index == dl.index[key]
			assert dl.batch_id[key] == 0
			assert dl.batch_size[key] == 3
			dl.restart(key, shuffle=True)
			assert dl.batch_id[key] == 0
			record_index = copy.copy(dl.index[key])
			dl.restart(key, shuffle=False)
			assert record_index == dl.index[key]
			assert dl.batch_id[key] == 0

	def base_test_get_batch(self, dl):
		with pytest.raises(ValueError):
			dl.get_batch("unknown set", [0, 1])
		for key in dl.key_name:
			with pytest.raises(IndexError):
				length = len(dl.data[key]['session'])
				dl.get_batch(key, [length-1, length])
			batch = dl.get_batch(key, [0, 1])
			assert len(dl.index[key]) >= 2
			assert len(batch["turn_length"]) == 2
			assert len(batch["sent_length"]) == 2
			assert isinstance(batch['sent_length'][0], list)
			assert batch['sent'].shape[0] == 2
			assert batch['sent'].shape[1] == max(batch['turn_length'])
			assert batch['sent'].shape[2] == max(chain(*batch['sent_length']))

			for idx in [0, 1]:
				for turn in range(batch['turn_length'][idx]):
					if batch['sent_length'][idx][turn] < batch['sent'].shape[2]:
						assert batch['sent'][idx][turn][batch['sent_length'][idx][turn]-1] == dl.eos_id
					assert batch['sent'][idx][turn][0] == dl.go_id

		# this is true, only when there is no unknown words in dl
		# (Only valid & invalid words)
		flag = False
		for key in dl.key_name:
			length = len(dl.data[key]['session'])
			for i in range(length):
				batch = dl.get_batch(key, [i])
				assert dl.unk_id not in batch["sent_allvocabs"]
				batch = dl.get_batch(key, [i])
				if dl.unk_id in batch["sent"]:
					flag = True
		assert flag

	def base_test_get_next_batch(self, dl):
		with pytest.raises(ValueError):
			dl.get_next_batch("unknown set")

		for key in dl.key_name:
			with pytest.raises(RuntimeError):
				dl.get_next_batch(key)

			dl.restart(key, 7)
			sample_num = 0
			while True:
				batch = dl.get_next_batch(key, ignore_left_samples=True)
				if not batch:
					break
				assert len(batch["turn_length"]) == 7
				assert len(batch["sent_length"]) == 7
				assert batch['sent'].shape[0] == 7
				sample_num += 7
			assert sample_num + 7 >= len(dl.data[key]['session'])

			dl.restart(key, 7)
			sample_num = 0
			while True:
				batch = dl.get_next_batch(key)
				assert batch is not None # dummy dataset must not be multiple of 7
				if len(batch["turn_length"]) == 7:
					assert len(batch["sent_length"]) == 7
					assert batch['sent'].shape[0] == 7
					sample_num += 7
				else:
					assert len(batch["sent_length"]) == batch['sent'].shape[0]
					assert len(batch["turn_length"]) == batch['sent'].shape[0]
					sample_num += batch['sent'].shape[0]
					batch = dl.get_next_batch(key)
					assert not batch
					break
			assert sample_num == len(dl.data[key]['session'])

	def base_test_convert(self, dl):
		sent_id = [0, 1, 2]
		sent = ["<pad>", "<unk>", "<go>"]
		assert sent == dl.index_to_sen(sent_id)
		assert sent_id == dl.sen_to_index(sent)

		sent = ["<unk>", "<go>", "<pad>", "<unkownword>", "<pad>", "<go>"]
		sent_id = [1, 2, 0, 1, 0, 2]
		assert sent_id == dl.sen_to_index(sent)
		assert sent_id == dl.sen_to_index(sent, invalid_vocab=True)

		sent = [dl.all_vocab_list[dl.vocab_size]]
		assert [1] == dl.sen_to_index(sent)
		assert [dl.vocab_size] == dl.sen_to_index(sent, invalid_vocab=True)

		sent_id = [0, 1, 2, 3, 0, 1, 0, 0]
		sent = ["<pad>", "<unk>", "<go>", "<eos>", "<pad>", "<unk>", "<pad>", "<pad>"]
		assert sent == dl.index_to_sen(sent_id, trim=False)
		sent = ["<pad>", "<unk>", "<go>"]
		assert sent == dl.index_to_sen(sent_id)

		sent_id = [0, 0, 3]
		sent = ["<pad>", "<pad>", "<eos>"]
		assert sent == dl.index_to_sen(sent_id, trim=False)
		assert not dl.index_to_sen(sent_id)

		sent_id = [3, 3, 3]
		sent = ["<eos>", "<eos>", "<eos>"]
		assert sent == dl.index_to_sen(sent_id, trim=False)
		assert not dl.index_to_sen(sent_id)

		sent_id = [0, 0, 0]
		sent = ["<pad>", "<pad>", "<pad>"]
		assert sent == dl.index_to_sen(sent_id, trim=False)
		assert not dl.index_to_sen(sent_id)

	def base_test_multi_turn_convert(self, dl):
		sent_id = [[0, 1, 2], [2, 1, 1]]
		sent = [["<pad>", "<unk>", "<go>"], ["<go>", "<unk>", "<unk>"]]
		assert sent == dl.multi_turn_index_to_sen(sent_id)
		assert sent_id == dl.multi_turn_sen_to_index(sent)

		sent = [["<unk>", "<go>", "<pad>", "<unkownword>", "<pad>", "<go>"], ["<go>", "<eos>"]]
		sent_id = [[1, 2, 0, 1, 0, 2], [2, 3]]
		assert sent_id == dl.multi_turn_sen_to_index(sent)

		sent_id = [[0, 1, 2, 2, 0, 3, 1, 0, 0], [0, 3, 2], [1, 2, 2, 0], [1, 2, 2, 3]]
		sent = [["<pad>", "<unk>", "<go>", "<go>", "<pad>", "<eos>", "<unk>", "<pad>", "<pad>"], \
				["<pad>", "<eos>", "<go>"], \
				["<unk>", "<go>", "<go>", "<pad>"], \
				["<unk>", "<go>", "<go>", "<eos>"]]
		assert sent == dl.multi_turn_index_to_sen(sent_id, trim=False)
		sent = [["<pad>", "<unk>", "<go>", "<go>"]]
		assert sent == dl.multi_turn_index_to_sen(sent_id)

		sent = [[dl.all_vocab_list[dl.vocab_size]]]
		assert [[1]] == dl.multi_turn_sen_to_index(sent)
		assert [[dl.vocab_size]] == dl.multi_turn_sen_to_index(sent, invalid_vocab=True)

	def base_test_teacher_forcing_metric(self, dl):
		assert isinstance(dl.get_teacher_forcing_metric(), MetricBase)

	def base_test_teacher_inference_metric(self, dl):
		assert isinstance(dl.get_inference_metric(), MetricBase)

	def base_test_multi_runs(self, dl_list):
		assert all(x.vocab_list == dl_list[0].vocab_list for x in dl_list)

	def base_test_hash(self, dl):
		recorder1 = HashValueRecorder()
		recorder2 = HashValueRecorder()

		for key in dl.key_name:
			dl.restart(key, 7)
			recorder1 = HashValueRecorder()
			while True:
				batch = dl.get_next_batch(key, needhash=True)
				if not batch:
					break
				recorder1.forward(batch)

			dl.restart(key, 7)
			recorder2 = HashValueRecorder()
			while True:
				batch = dl.get_next_batch(key, needhash=True)
				if not batch:
					break
				recorder2.forward(batch)

			assert recorder1.close()['hashvalue'] == recorder2.close()['hashvalue'] 

@pytest.fixture
def load_ubuntucorpus():
	def _load_ubuntucorpus(invalid_vocab_times=0):
		return UbuntuCorpus("./tests/dataloader/dummy_ubuntucorpus", invalid_vocab_times=invalid_vocab_times)
	return _load_ubuntucorpus

class TestUbuntuCorpus(TestMultiTurnDialog):

	@pytest.mark.dependency()
	def test_init(self, load_ubuntucorpus):
		super().base_test_init(load_ubuntucorpus())

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

	@pytest.mark.dependency(depends=["TestUbuntuCorpus::test_init"])
	def test_hash(self, load_ubuntucorpus):
		super().base_test_hash(load_ubuntucorpus())

@pytest.fixture
def load_switchboardcorpus():
	def _load_switchboardcorpus(invalid_vocab_times=0):
		return SwitchboardCorpus("./tests/dataloader/dummy_switchboardcorpus", invalid_vocab_times=invalid_vocab_times)
	return _load_switchboardcorpus

class TestSwitchboardCorpus(TestMultiTurnDialog):

	@pytest.mark.dependency()
	def test_init(self, load_switchboardcorpus):
		super().base_test_init(load_switchboardcorpus())

	def test_restart(self, load_switchboardcorpus):
		super().base_test_restart(load_switchboardcorpus())

	@pytest.mark.dependency(depends=["TestSwitchboardCorpus::test_init"])
	def test_get_batch(self, load_switchboardcorpus):
		super().base_test_get_batch(load_switchboardcorpus())

	@pytest.mark.dependency(depends=["TestSwitchboardCorpus::test_init"])
	def test_get_next_batch(self, load_switchboardcorpus):
		super().base_test_get_next_batch(load_switchboardcorpus())

	@pytest.mark.dependency(depends=["TestSwitchboardCorpus::test_init"])
	def test_convert(self, load_switchboardcorpus):
		super().base_test_convert(load_switchboardcorpus())

	def test_multi_turn_convert(self, load_switchboardcorpus):
		super().base_test_multi_turn_convert(load_switchboardcorpus())

	def test_teacher_forcing_metric(self, load_switchboardcorpus):
		super().base_test_teacher_forcing_metric(load_switchboardcorpus())

	def test_teacher_inference_metric(self, load_switchboardcorpus):
		super().base_test_teacher_inference_metric(load_switchboardcorpus())

	def test_init_multi_runs(self, load_switchboardcorpus):
		super().base_test_multi_runs([load_switchboardcorpus() for i in range(3)])

	@pytest.mark.skip()
	@pytest.mark.dependency(depends=["TestSwitchboardCorpus::test_init"])
	def test_hash(self, load_switchboardcorpus):
		super().base_test_hash(load_switchboardcorpus())

	# TODO: add test for get_batch
	# TODO: add test for inference metric
