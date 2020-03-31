from collections import OrderedDict

import pytest

import sys
sys.path.append('/home/zhengchujie/GitHub/cotk')

from cotk.dataloader import Vocab, GeneralVocab, PretrainedVocab, SimpleVocab, PretrainedTokenizer

def setup_module():
	import random
	random.seed(0)
	import numpy as np
	np.random.seed(0)

class TestVocab():
	def base_test_init(self):
		with pytest.raises(NotImplementedError):
			Vocab()
		
		class DummyVocab(Vocab):
			def __init__(self):
				super().__init__()
		assert DummyVocab()._setting_hash is None
		
		with pytest.raises(NotImplementedError):
			DummyVocab().add_tokens(['a'], 'b')
		with pytest.raises(NotImplementedError):
			DummyVocab().build_vocab()
		with pytest.raises(NotImplementedError):
			DummyVocab().convert_tokens_to_ids(['a'])
		with pytest.raises(NotImplementedError):
			DummyVocab().convert_ids_to_tokens([0])
			
		with pytest.raises(NotImplementedError):
			DummyVocab().frequent_vocab_size
		with pytest.raises(NotImplementedError):
			DummyVocab().all_vocab_size
		with pytest.raises(NotImplementedError):
			DummyVocab().frequent_vocab_list
		with pytest.raises(NotImplementedError):
			DummyVocab().all_vocab_list
			
		with pytest.raises(NotImplementedError):
			DummyVocab().get_special_tokens_mapping()
		with pytest.raises(NotImplementedError):
			DummyVocab().pad_id
		with pytest.raises(NotImplementedError):
			DummyVocab().unk_id
		with pytest.raises(NotImplementedError):
			DummyVocab().go_id
		with pytest.raises(NotImplementedError):
			DummyVocab().eos_id
		
		dv = DummyVocab()
		dv._setting_hash = 'abc'
		assert dv.get_setting_hash() == 'abc'
		
		with pytest.raises(NotImplementedError):
			DummyVocab().get_vocab_hash()


@pytest.fixture()
def load_generalvocab():
	def _load_generalvocab(min_frequent_vocab_times=10,
						   min_rare_vocab_times=0,
						   special_tokens_mapping=None,
						   special_appeared_in_data=False):
		return GeneralVocab(min_frequent_vocab_times, min_rare_vocab_times,
							special_tokens_mapping, special_appeared_in_data)
	return _load_generalvocab

@pytest.fixture()
def load_predefined_generalvocab():
	def _load_predefined_generalvocab(vocab_list=None, frequent_vocab_size=20,
									  special_tokens_mapping=None):
		if vocab_list is None:
			vocab_list = ['<pad>', '<unk>', '<go>', '<eos>'] + [chr(i) for i in range(97, 123)]
		return GeneralVocab.from_predefined(vocab_list, frequent_vocab_size, special_tokens_mapping)
	return _load_predefined_generalvocab


@pytest.fixture()
def load_frequent_generalvocab():
	def _load_frequent_generalvocab(frequent_vocab_list=None,
									  special_tokens_mapping=None):
		if frequent_vocab_list is None:
			frequent_vocab_list = ['<pad>', '<unk>', '<go>', '<eos>'] + [chr(i) for i in range(97, 123)]
		return GeneralVocab.from_frequent_word(frequent_vocab_list, special_tokens_mapping)
	return _load_frequent_generalvocab


class TestGeneralVocab(TestVocab):
	@pytest.mark.dependency()
	def test_init(self, load_generalvocab):
		super().base_test_init()
		vocab = load_generalvocab()
		assert vocab.min_frequent_vocab_times == 10
		assert vocab.min_rare_vocab_times == 0
		assert vocab.special_appeared_in_data == False
		assert vocab.special_tokens_mapping == OrderedDict(
			[("pad", "<pad>"), ("unk", "<unk>"), ("go", "<go>"), ("eos", "<eos>")]
		)
		assert vocab.mode == 'init'
		assert vocab.train_tokens == []
		assert vocab.test_tokens == []
		assert vocab._all_vocab_list is None
		assert vocab.word2id is None
		assert vocab._frequent_vocab_size == 0
		assert vocab.get_setting_hash() == load_generalvocab().get_setting_hash()
	
	@pytest.mark.dependency()
	def test_from_predefined(self, load_predefined_generalvocab):
		with pytest.raises(ValueError):
			load_predefined_generalvocab(vocab_list=[chr(i) for i in range(97, 123)])
		with pytest.raises(ValueError):
			load_predefined_generalvocab(vocab_list=['<pad>', '<unk>', '<go>', '<eos>'] + 2 * [chr(i) for i in range(97, 123)])
		
		vocab_list = ['<pad>', '<unk>', '<go>', '<eos>'] + [chr(i) for i in range(97, 123)]
		vocab = load_predefined_generalvocab(vocab_list=vocab_list, frequent_vocab_size=20)
		assert vocab.mode == 'finish'
		assert vocab.all_vocab_list == vocab_list
		assert vocab.frequent_vocab_size == 20
		assert vocab.word2id == {w: i for i, w in enumerate(vocab_list)}
		assert vocab.train_tokens is None
		assert vocab.test_tokens is None
		assert vocab.get_setting_hash() == load_predefined_generalvocab(vocab_list=vocab_list, frequent_vocab_size=20).get_setting_hash()
		
		assert vocab.all_vocab_size == len(vocab.all_vocab_list)
		assert vocab.frequent_vocab_size == len(vocab.frequent_vocab_list)
	
	@pytest.mark.dependency()
	def test_from_predefined_vocab(self, load_predefined_generalvocab):
		with pytest.raises(TypeError):
			GeneralVocab.from_predefined_vocab([])
		
		vocab_list = ['<pad>', '<unk>', '<go>', '<eos>'] + [chr(i) for i in range(97, 123)]
		vocab = load_predefined_generalvocab(vocab_list=vocab_list, frequent_vocab_size=20)
		vocab2 = GeneralVocab.from_predefined_vocab(vocab)
		assert vocab.all_vocab_list == vocab2.all_vocab_list
		assert vocab.all_vocab_size == vocab2.all_vocab_size
		assert vocab.frequent_vocab_size == vocab2.frequent_vocab_size
	
	@pytest.mark.dependency()
	def test_from_frequent_word(self, load_frequent_generalvocab):
		with pytest.raises(ValueError):
			load_frequent_generalvocab(frequent_vocab_list=[chr(i) for i in range(97, 123)])
		
		frequent_vocab_list = ['<pad>', '<unk>', '<go>', '<eos>'] + [chr(i) for i in range(97, 123)]
		vocab = load_frequent_generalvocab(frequent_vocab_list=frequent_vocab_list)
		assert vocab.mode == "frequent_specified"
		assert vocab.all_vocab_list == frequent_vocab_list
		assert vocab.get_setting_hash() == load_frequent_generalvocab(frequent_vocab_list).get_setting_hash()
	
	@pytest.mark.dependency()
	def test_from_frequent_word_of_vocab(self, load_predefined_generalvocab):
		with pytest.raises(TypeError):
			GeneralVocab.from_frequent_word_of_vocab([])
		
		vocab_list = ['<pad>', '<unk>', '<go>', '<eos>'] + [chr(i) for i in range(97, 123)]
		vocab = load_predefined_generalvocab(vocab_list=vocab_list)
		vocab2 = GeneralVocab.from_frequent_word_of_vocab(vocab)
		assert vocab2.mode == 'finish'
		assert vocab2.all_vocab_list == vocab.frequent_vocab_list
		assert vocab2.all_vocab_size == vocab.frequent_vocab_size
		assert vocab2.get_special_tokens_mapping() == vocab.get_special_tokens_mapping()
	
	@pytest.mark.dependency()
	def test_add_tokens(self, load_generalvocab, load_predefined_generalvocab):
		vocab = load_predefined_generalvocab()
		vocab.add_tokens([], 'any_str')
		assert vocab.train_tokens is None and vocab.test_tokens is None
		
		vocab = load_generalvocab()
		train_tokens = [e for i in range(97, 123) for e in [chr(i)] * (i-96)]
		test_tokens = [e for i in range(110, 123) for e in [chr(i)] * (i-100)]
		vocab.add_tokens(train_tokens, 'train')
		assert vocab.train_tokens[-len(train_tokens):] == train_tokens
		vocab.add_tokens(test_tokens, 'test')
		assert vocab.test_tokens[-len(test_tokens):] == test_tokens
		
		tmp_train_tokens = vocab.train_tokens[:]
		tmp_test_tokens = vocab.test_tokens[:]
		vocab.add_tokens(test_tokens, 'extra')
		assert tmp_train_tokens == vocab.train_tokens
		assert tmp_test_tokens == vocab.test_tokens
		vocab.add_tokens(test_tokens, 'train')
		assert tmp_train_tokens + test_tokens == vocab.train_tokens
		
		with pytest.raises(ValueError):
			vocab = load_generalvocab(min_frequent_vocab_times=10)
			train_tokens = [e for i in range(97, 123) for e in [chr(i)] * (i-96)]
			vocab.add_tokens(train_tokens, 'unknown')
		
	@pytest.mark.dependency()
	def test_build_vocab(self, load_generalvocab, load_predefined_generalvocab, load_frequent_generalvocab):
		vocab = load_predefined_generalvocab()
		vocab.build_vocab()
		
		with pytest.raises(RuntimeError):
			vocab = load_generalvocab()
			vocab.train_tokens = None
			vocab.build_vocab()
		
		with pytest.raises(RuntimeError):
			vocab = load_generalvocab(special_appeared_in_data=False)
			train_tokens = ['<pad>', '<unk>'] + [e for i in range(97, 123) for e in [chr(i)] * (i-96)]
			vocab.add_tokens(train_tokens, 'train')
			vocab.build_vocab()
		
		vocab = load_generalvocab(min_frequent_vocab_times=10)
		train_tokens = [e for i in range(97, 123) for e in [chr(i)] * (i-96)]
		test_tokens = [e for i in range(110, 123) for e in [chr(i)] * (i-100)]
		vocab.add_tokens(train_tokens, 'train')
		vocab.add_tokens(test_tokens, 'test')
		vocab.build_vocab()
		
		frequent_vocab = [chr(i) for i in range(97, 123) if i-96 >= 10]
		frequent_vocab.reverse()
		rare_vocab = [chr(i) for i in range(97, 123) if i-96 < 10]
		rare_vocab.reverse()
		special_tokens = list(vocab.special_tokens_mapping.values())
		assert vocab.frequent_vocab_list == special_tokens + frequent_vocab
		assert vocab.all_vocab_list == special_tokens + frequent_vocab + rare_vocab
		assert vocab.frequent_vocab_size == len(special_tokens) + len(frequent_vocab)
		assert vocab.train_tokens is None and vocab.test_tokens is None
		assert vocab.mode == 'finish'
	
	@pytest.mark.dependency()
	def test_get_special_tokens_id(self, load_predefined_generalvocab):
		vocab = load_predefined_generalvocab()
		assert vocab.get_special_tokens_id('pad') == vocab.pad_id
		assert vocab.get_special_tokens_id('unk') == vocab.unk_id
		assert vocab.get_special_tokens_id('go') == vocab.go_id
		assert vocab.get_special_tokens_id('eos') == vocab.eos_id
		
		with pytest.raises(KeyError):
			vocab.get_special_tokens_id('unknown_token')
	
	@pytest.mark.dependency()
	def test_convert(self, load_predefined_generalvocab):
		with pytest.raises(RuntimeError):
			vocab = GeneralVocab()
			vocab.convert_ids_to_tokens([0])
		with pytest.raises(RuntimeError):
			vocab = GeneralVocab()
			vocab.convert_tokens_to_ids([''])
			
		vocab = load_predefined_generalvocab()
		tokens = ['<pad>', '<unk>', '<go>', '<eos>'] + [chr(i) for i in range(97, 123)]
		assert vocab.convert_ids_to_tokens(vocab.convert_tokens_to_ids(tokens, only_frequent_word=True)) != tokens
		assert vocab.convert_ids_to_tokens(vocab.convert_tokens_to_ids(tokens, only_frequent_word=False)) == tokens
	
	@pytest.mark.dependency()
	def test_get_vocab_hash(self, load_predefined_generalvocab):
		assert load_predefined_generalvocab().get_vocab_hash() == \
			   load_predefined_generalvocab().get_vocab_hash()
		assert load_predefined_generalvocab(frequent_vocab_size=10).get_vocab_hash() == \
			   load_predefined_generalvocab(frequent_vocab_size=10).get_vocab_hash()
		assert load_predefined_generalvocab(frequent_vocab_size=10).get_vocab_hash() != \
			   load_predefined_generalvocab(frequent_vocab_size=20).get_vocab_hash()

@pytest.fixture
def load_PretrainedVocab():
	def _load_PretrainedVocab():
		vocab_file = './tests/dataloader/dummy_gpt2vocab/vocab.json'
		merges_file = './tests/dataloader/dummy_gpt2vocab/merges.txt'
		from transformers.tokenization_gpt2 import GPT2Tokenizer
		return PretrainedVocab(GPT2Tokenizer(vocab_file, merges_file, unk_token='<|endoftext|>'))
	return _load_PretrainedVocab


class TestPretrainedVocab():
	@pytest.mark.dependency()
	def test_init(self, load_PretrainedVocab):
		vocab = load_PretrainedVocab()
		vocab_file = './tests/dataloader/dummy_gpt2vocab/vocab.json'
		merges_file = './tests/dataloader/dummy_gpt2vocab/merges.txt'
		from transformers.tokenization_gpt2 import GPT2Tokenizer
		toker = PretrainedTokenizer(GPT2Tokenizer(vocab_file, merges_file, unk_token='<|endoftext|>'))
		assert vocab.tokenizer.get_setting_hash() == toker.get_setting_hash()
		assert vocab.get_setting_hash() == load_PretrainedVocab().get_setting_hash()
		assert vocab.get_vocab_hash() == load_PretrainedVocab().get_vocab_hash()
	
	@pytest.mark.dependency()
	def test_convert(self, load_PretrainedVocab):
		vocab = load_PretrainedVocab()
		vocab_file = './tests/dataloader/dummy_gpt2vocab/vocab.json'
		merges_file = './tests/dataloader/dummy_gpt2vocab/merges.txt'
		from transformers.tokenization_gpt2 import GPT2Tokenizer
		toker = GPT2Tokenizer(vocab_file, merges_file, unk_token='<|endoftext|>')
		assert vocab.frequent_vocab_size == vocab.all_vocab_size == len(toker.encoder)
		assert vocab.frequent_vocab_list == vocab.all_vocab_list == list(map(lambda x: x[0], sorted(toker.encoder.items(), key=lambda x: x[1])))
		
		tokens = ['A', 'Ġbeautiful', 'Ġdessert', 'Ġwaiting', 'Ġto', 'Ġbe', 'Ġshared', 'Ġby', 'Ġtwo', 'Ġpeople', '.']
		assert vocab.convert_ids_to_tokens(vocab.convert_tokens_to_ids(tokens, False)) == tokens
		assert vocab.convert_ids_to_tokens(vocab.convert_tokens_to_ids(tokens, True)) == tokens

	@pytest.mark.dependency()
	def test_special_tokens(self, load_PretrainedVocab):
		vocab = load_PretrainedVocab()
		special_tokens_mapping = vocab.get_special_tokens_mapping()
		for key, value in special_tokens_mapping.items():
			assert vocab.convert_tokens_to_ids([value])[0] == vocab.get_special_tokens_id(key)
		with pytest.raises(KeyError):
			vocab.get_special_tokens_id('unknown_special')

@pytest.fixture()
def load_SimpleVocab():
	def _load_SimpleVocab():
		vocab = SimpleVocab()
		tokens = [e for i in range(97, 123) for e in [chr(i)] * (i-96)]
		vocab.add_tokens(tokens, '')
		vocab.build_vocab()
		return vocab
	return _load_SimpleVocab

class TestSimpleVocab():
	@pytest.mark.dependency()
	def test_init(self, load_SimpleVocab):
		vocab = SimpleVocab()
		assert vocab.get_setting_hash() == SimpleVocab().get_setting_hash()
		assert len(vocab._token_counter) == 0
		assert vocab._all_vocab_list is None
		assert vocab.word2id is None
		assert vocab.mode == 'init'
	
	@pytest.mark.dependency()
	def test_add_tokens(self, load_SimpleVocab):
		vocab = SimpleVocab()
		tokens = [e for i in range(97, 123) for e in [chr(i)] * (i-96)]
		vocab.add_tokens(tokens, '')
		for i in range(97, 123):
			assert vocab._token_counter[chr(i)] == (i-96)
		vocab.add_tokens(tokens, '')
		for i in range(97, 123):
			assert vocab._token_counter[chr(i)] == 2 * (i-96)
	
	@pytest.mark.dependency()
	def test_build_vocab(self, load_SimpleVocab):
		vocab = SimpleVocab()
		tokens = [e for i in range(97, 123) for e in [chr(i)] * (i-96)]
		vocab.add_tokens(tokens, '')
		
		vocab.mode = 'finish'
		vocab.build_vocab()
		
		vocab.mode = 'init'
		vocab.build_vocab()
		vocab_list = [chr(i) for i in range(97, 123)]
		vocab_list.reverse()
		assert vocab.frequent_vocab_size == vocab.all_vocab_size == len(vocab_list)
		assert vocab.frequent_vocab_list == vocab.all_vocab_list == vocab_list
		assert vocab.mode == 'finish'
		assert vocab._token_counter is None
	
	@pytest.mark.dependency()
	def test_convert(self, load_SimpleVocab):
		vocab = SimpleVocab()
		with pytest.raises(RuntimeError):
			vocab.convert_tokens_to_ids([''])
		with pytest.raises(RuntimeError):
			vocab.convert_ids_to_tokens([0])
		tokens = [e for i in range(97, 123) for e in [chr(i)] * (i-96)]
		vocab.add_tokens(tokens, '')
		vocab.build_vocab()
		
		tokens = [chr(i) for i in range(97, 123)]
		assert vocab.convert_ids_to_tokens(vocab.convert_tokens_to_ids(tokens, only_frequent_word=True)) == tokens
		assert vocab.convert_ids_to_tokens(vocab.convert_tokens_to_ids(tokens, only_frequent_word=False)) == tokens
	
	@pytest.mark.dependency()
	def test_special_tokens(self, load_SimpleVocab):
		vocab = load_SimpleVocab()
		assert vocab.get_special_tokens_mapping() == {}
		with pytest.raises(NotImplementedError):
			vocab.get_special_tokens_id('')
	
	@pytest.mark.dependency()
	def test_get_vocab_hash(self, load_SimpleVocab):
		vocab = load_SimpleVocab()
		assert vocab.get_vocab_hash() == load_SimpleVocab().get_vocab_hash()
		
		vocab2 = SimpleVocab()
		tokens = [e for i in range(97, 113) for e in [chr(i)] * (i-96)]
		vocab2.add_tokens(tokens, '')
		vocab2.build_vocab()
		assert vocab.get_vocab_hash() != vocab2.get_vocab_hash()