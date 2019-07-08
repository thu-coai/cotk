import copy

import pytest
from pytest_mock import mocker

from cotk.dataloader import SingleTurnDialog, OpenSubtitles
from cotk.metric import MetricBase

def setup_module():
	import random
	random.seed(0)
	import numpy as np
	np.random.seed(0)

class TestSingleTurnDialog():
	def base_test_init(self, dl):
		assert isinstance(dl, SingleTurnDialog)
		assert isinstance(dl.ext_vocab, list)
		assert dl.ext_vocab[:4] == ["<pad>", "<unk>", "<go>", "<eos>"]
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
		for key in dl.key_name:
			post = dl.data[key]['post']
			resp = dl.data[key]['resp']
			assert len(post) == len(resp)
			assert isinstance(post[0], list)
			assert isinstance(resp[0], list)
			assert post[0][0] == dl.go_id
			assert post[0][-1] == dl.eos_id
			assert resp[0][0] == dl.go_id
			assert resp[0][-1] == dl.eos_id

		# assert the data has valid token
		assert dl.vocab_size > 4
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
				length = len(dl.data[key]['post'])
				dl.get_batch(key, [length-1, length])
			assert len(dl.index[key]) >= 2
			batch = dl.get_batch(key, [0, 1])
			assert len(batch["post_length"]) == 2
			assert len(batch["resp_length"]) == 2
			assert batch["post"].shape[0] == 2
			assert batch["resp"].shape[0] == 2

			for sent, length in [("post", "post_length"), ("resp", "resp_length")]:
				for idx in [0, 1]:
					if batch[length][idx] < batch[sent].shape[1]:
						assert batch[sent][idx][batch[length][idx]-1] == dl.eos_id
					assert batch[sent][idx][0] == dl.go_id

		# this is true, only when there is no unknown words in dl
		# (Only valid & invalid words)
		flag = False
		for key in dl.key_name:
			length = len(dl.data[key]['post'])
			for i in range(length):
				batch = dl.get_batch(key, [i])
				assert dl.unk_id not in batch["post_allvocabs"]
				assert dl.unk_id not in batch["resp_allvocabs"]
				batch = dl.get_batch(key, [i])
				if dl.unk_id in batch["post"] or \
					dl.unk_id in batch["resp"]:
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
				assert batch["post"].shape[0] == 7
				sample_num += batch["post"].shape[0]
			assert sample_num + 7 >= len(dl.data[key]["post"])

			dl.restart(key, 7)
			sample_num = 0
			while True:
				batch = dl.get_next_batch(key)
				assert batch is not None # dummy dataset must not be multiple of 7
				if batch["post"].shape[0] == 7:
					sample_num += 7
				else:
					sample_num += batch["post"].shape[0]
					batch = dl.get_next_batch(key)
					assert not batch
					break
			assert sample_num == len(dl.data[key]["post"])

	def base_test_convert(self, dl):
		sent_id = [0, 1, 2]
		sent = ["<pad>", "<unk>", "<go>"]
		assert sent == dl.convert_ids_to_tokens(sent_id)
		assert sent_id == dl.convert_tokens_to_ids(sent)

		sent = ["<unk>", "<go>", "<pad>", "<unkownword>", "<pad>", "<go>"]
		sent_id = [1, 2, 0, 1, 0, 2]
		assert sent_id == dl.convert_tokens_to_ids(sent)
		assert sent_id == dl.convert_tokens_to_ids(sent, invalid_vocab=True)

		sent = [dl.all_vocab_list[dl.vocab_size]]
		assert [1] == dl.convert_tokens_to_ids(sent)
		assert [dl.vocab_size] == dl.convert_tokens_to_ids(sent, invalid_vocab=True)


		sent_id = [0, 1, 2, 0, 0, 3, 1, 0, 0]
		sent = ["<pad>", "<unk>", "<go>", "<pad>", "<pad>", "<eos>", "<unk>", "<pad>", "<pad>"]
		assert sent == dl.convert_ids_to_tokens(sent_id, trim=False)
		sent = ["<pad>", "<unk>", "<go>"]
		assert sent == dl.convert_ids_to_tokens(sent_id)

		sent_id = [0, 0, 3]
		sent = ["<pad>", "<pad>", "<eos>"]
		assert sent == dl.convert_ids_to_tokens(sent_id, trim=False)
		assert not dl.convert_ids_to_tokens(sent_id)

		sent_id = [3, 3, 3]
		sent = ["<eos>", "<eos>", "<eos>"]
		assert sent == dl.convert_ids_to_tokens(sent_id, trim=False)
		assert not dl.convert_ids_to_tokens(sent_id)

		sent_id = [0, 0, 0]
		sent = ["<pad>", "<pad>", "<pad>"]
		assert sent == dl.convert_ids_to_tokens(sent_id, trim=False)
		assert not dl.convert_ids_to_tokens(sent_id)

	def base_test_teacher_forcing_metric(self, dl):
		assert isinstance(dl.get_teacher_forcing_metric(), MetricBase)

	def base_test_teacher_inference_metric(self, dl):
		assert isinstance(dl.get_inference_metric(), MetricBase)

	def base_test_multi_runs(self, dl_list):
		assert all(x.vocab_list == dl_list[0].vocab_list for x in dl_list)

@pytest.fixture
def load_opensubtitles():
	def _load_opensubtitles(invalid_vocab_times=0):
		return OpenSubtitles("./tests/dataloader/dummy_opensubtitles", invalid_vocab_times=invalid_vocab_times)
	return _load_opensubtitles

class TestOpenSubtitles(TestSingleTurnDialog):

	@pytest.mark.dependency()
	def test_init(self, load_opensubtitles):
		super().base_test_init(load_opensubtitles())
		super().base_test_all_unknown(load_opensubtitles(10000))

	def test_restart(self, load_opensubtitles):
		super().base_test_restart(load_opensubtitles())

	@pytest.mark.dependency(depends=["TestOpenSubtitles::test_init"])
	def test_get_batch(self, load_opensubtitles):
		super().base_test_get_batch(load_opensubtitles())

	@pytest.mark.dependency(depends=["TestOpenSubtitles::test_init"])
	def test_get_next_batch(self, load_opensubtitles):
		super().base_test_get_next_batch(load_opensubtitles())

	@pytest.mark.dependency(depends=["TestOpenSubtitles::test_init"])
	def test_convert(self, load_opensubtitles):
		super().base_test_convert(load_opensubtitles())

	def test_teacher_forcing_metric(self, load_opensubtitles):
		super().base_test_teacher_forcing_metric(load_opensubtitles())

	def test_teacher_inference_metric(self, load_opensubtitles):
		super().base_test_teacher_inference_metric(load_opensubtitles())

	def test_init_multi_runs(self, load_opensubtitles):
		super().base_test_multi_runs([load_opensubtitles() for i in range(3)])

