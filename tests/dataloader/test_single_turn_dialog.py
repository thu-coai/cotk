import copy

import pytest

from contk.dataloader import SingleTurnDialog, OpenSubtitles
from contk.metric import MetricBase

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
		assert isinstance(dl.vocab_list, list)
		assert dl.vocab_list[:len(dl.ext_vocab)] == dl.ext_vocab
		assert isinstance(dl.word2id, dict)
		assert len(dl.word2id) == len(dl.vocab_list)
		for i, word in enumerate(dl.vocab_list):
			assert isinstance(word, str)
			assert dl.word2id[word] == i
		assert dl.vocab_size == len(dl.vocab_list)
		post = dl.data["train"]['post']
		resp = dl.data["train"]['resp']
		assert len(post) == len(resp)
		assert isinstance(post[0], list)
		assert isinstance(resp[0], list)
		assert post[0][0] == dl.go_id
		assert post[0][-1] == dl.eos_id
		assert resp[0][0] == dl.go_id
		assert resp[0][-1] == dl.eos_id

	def base_test_restart(self, dl):
		with pytest.raises(ValueError):
			dl.restart("unknown set")
		with pytest.raises(ValueError):
			dl.restart("train")
		record_index = copy.copy(dl.index["train"])
		dl.restart("train", batch_size=3, shuffle=False)
		assert record_index == dl.index['train']
		assert dl.batch_id["train"] == 0
		assert dl.batch_size["train"] == 3
		dl.restart("train", shuffle=True)
		assert dl.batch_id["train"] == 0
		record_index = copy.copy(dl.index["train"])
		dl.restart("train", shuffle=False)
		assert record_index == dl.index['train']
		assert dl.batch_id["train"] == 0

	def base_test_get_batch(self, dl):
		assert len(dl.index["train"]) >= 2
		batch = dl.get_batch("train", [0, 1])
		assert len(batch["post_length"]) == 2
		assert len(batch["resp_length"]) == 2
		assert batch["post"].shape[0] == 2
		assert batch["resp"].shape[0] == 2

	def base_test_get_next_batch(self, dl):
		with pytest.raises(ValueError):
			dl.get_next_batch("unknown set")
		with pytest.raises(RuntimeError):
			dl.get_next_batch("train")

		dl.restart("train", 7)
		sample_num = 0
		while True:
			batch = dl.get_next_batch("train", ignore_left_samples=True)
			if not batch:
				break
			assert batch["post"].shape[0] == 7
			sample_num += batch["post"].shape[0]
		assert sample_num + 7 >= len(dl.data["train"]["post"])

		dl.restart("train", 7)
		sample_num = 0
		while True:
			batch = dl.get_next_batch("train")
			if not batch:
				break
			sample_num += batch["post"].shape[0]
		assert sample_num == len(dl.data["train"]["post"])

	def base_test_convert(self, dl):
		sent_id = [1, 2, 3]
		sent = ["<unk>", "<go>", "<eos>"]
		assert sent == dl.index_to_sen(sent_id)
		assert sent_id == dl.sen_to_index(sent)

		sent = ["<unk>", "<go>", "<eos>", "<unkownword>", "<pad>"]
		sent_id = [1, 2, 3, 1, 0]
		assert sent_id == dl.sen_to_index(sent)

		sent_id = [1, 2, 3, 1, 0]
		sent = ["<unk>", "<go>", "<eos>", "<unk>", "<pad>"]
		assert sent == dl.index_to_sen(sent_id, trim=False)
		sent = ["<unk>", "<go>", "<eos>", "<unk>"]
		assert sent == dl.index_to_sen(sent_id)

	def base_test_teacher_forcing_metric(self, dl):
		assert isinstance(dl.get_teacher_forcing_metric(), MetricBase)

	def base_test_teacher_inference_metric(self, dl):
		assert isinstance(dl.get_inference_metric(), MetricBase)

@pytest.fixture
def load_opensubtitles():
	return OpenSubtitles("./tests/dataloader/dummy_opensubtitles")

class TestOpenSubtitles(TestSingleTurnDialog):
	def test_init(self, load_opensubtitles):
		super().base_test_init(load_opensubtitles)

	def test_restart(self, load_opensubtitles):
		super().base_test_restart(load_opensubtitles)

	def test_get_batch(self, load_opensubtitles):
		super().base_test_get_batch(load_opensubtitles)

	def test_get_next_batch(self, load_opensubtitles):
		super().base_test_get_next_batch(load_opensubtitles)

	def test_convert(self, load_opensubtitles):
		super().base_test_convert(load_opensubtitles)
