import copy

import pytest

from contk.dataloader import LanguageGeneration, MSCOCO
from contk.metric import MetricBase

class TestLanguageGeneration():
	def base_test_init(self, dl):
		assert isinstance(dl, LanguageGeneration)
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
		sentence = dl.data["train"]['sen']
		assert isinstance(sentence, list)
		assert isinstance(sentence[0], list)
		assert sentence[0][0] == dl.go_id
		assert sentence[0][-1] == dl.eos_id

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
		assert len(batch["sentence_length"]) == 2
		assert batch["sentence"].shape[0] == 2

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
			assert batch["sentence"].shape[0] == 7
			sample_num += batch["sentence"].shape[0]
		assert sample_num + 7 >= len(dl.data["train"]['sen'])

		dl.restart("train", 7)
		sample_num = 0
		while True:
			batch = dl.get_next_batch("train")
			if not batch:
				break
			sample_num += batch["sentence"].shape[0]
		assert sample_num == len(dl.data["train"]['sen'])

	def base_test_convert(self, dl):
		sent_id = [0, 1, 2]
		sent = ["<pad>", "<unk>", "<go>"]
		assert sent == dl.index_to_sen(sent_id)
		assert sent_id == dl.sen_to_index(sent)

		sent = ["<unk>", "<go>", "<pad>", "<unkownword>", "<pad>", "<go>"]
		sent_id = [1, 2, 0, 1, 0, 2]
		assert sent_id == dl.sen_to_index(sent)

		sent_id = [0, 1, 2, 0, 0, 3, 1, 0, 0]
		sent = ["<pad>", "<unk>", "<go>", "<pad>", "<pad>", "<eos>", "<unk>", "<pad>", "<pad>"]
		assert sent == dl.index_to_sen(sent_id, trim=False)
		sent = ["<pad>", "<unk>", "<go>"]
		assert sent == dl.index_to_sen(sent_id)

	def base_test_teacher_forcing_metric(self, dl):
		assert isinstance(dl.get_teacher_forcing_metric(), MetricBase)

	def base_test_teacher_inference_metric(self, dl):
		assert isinstance(dl.get_inference_metric(), MetricBase)

	def base_test_multi_runs(self, dl_list):
		assert all(x.vocab_list == dl_list[0].vocab_list for x in dl_list)

@pytest.fixture
def load_mscoco():
	def _load_mscoco():
		return MSCOCO("./tests/dataloader/dummy_mscoco")
	return _load_mscoco

class TestMSCOCO(TestLanguageGeneration):
	def test_init(self, load_mscoco):
		super().base_test_init(load_mscoco())

	def test_restart(self, load_mscoco):
		super().base_test_restart(load_mscoco())

	def test_get_batch(self, load_mscoco):
		super().base_test_get_batch(load_mscoco())

	def test_get_next_batch(self, load_mscoco):
		super().base_test_get_next_batch(load_mscoco())

	def test_convert(self, load_mscoco):
		super().base_test_convert(load_mscoco())

	def test_init_multi_runs(self, load_mscoco):
		super().base_test_multi_runs([load_mscoco() for i in range(3)])
