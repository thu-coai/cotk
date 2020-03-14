import copy
import random
import operator
from typing import List

import pytest
from pytest_mock import mocker
import numpy as np

from cotk.dataloader import Tokenizer, SimpleTokenizer, PretrainedTokenizer

def setup_module():
	import random
	random.seed(0)
	import numpy as np
	np.random.seed(0)

class TestTokenizer():
	def base_test_init(self):
		with pytest.raises(NotImplementedError):
			Tokenizer().tokenize('')
		
		with pytest.raises(NotImplementedError):
			Tokenizer().convert_tokens_to_sentence([''])
		
		with pytest.raises(NotImplementedError):
			Tokenizer().get_setting_hash()
	
	def base_test_tokenize_sentences(self, toker: Tokenizer, sentences: List[str]):
		tokenized_sentences = toker.tokenize_sentences(sentences)
		assert isinstance(tokenized_sentences, list)
		assert len(tokenized_sentences) == len(sentences)
		for sent in tokenized_sentences:
			assert isinstance(sent, list)
			for token in sent:
				assert isinstance(token, str)

	def base_test_tokenize_sessions(self, toker: Tokenizer, sessions: List[List[str]]):
		tokenized_sessions = toker.tokenize_sessions(sessions)
		assert isinstance(tokenized_sessions, list)
		assert len(tokenized_sessions) == len(sessions)
		for tk_session, session in zip(tokenized_sessions, sessions):
			assert isinstance(tk_session, list)
			assert len(tk_session) == len(session)
			tokenized_sentences = toker.tokenize_sentences(session)
			assert len(tk_session) == len(tokenized_sentences)
			for tk_session_sent, tk_sentence_sent in zip(tk_session, tokenized_sentences):
				assert tk_session_sent == tk_sentence_sent


@pytest.fixture
def load_SimpleTokenizer():
	def _load_SimpleTokenizer(method='space', special_tokens=None):
		if special_tokens is None:
			special_tokens = ['<pad>', '<unk>', '<go>', '<eos>']
		return SimpleTokenizer(method=method, special_tokens=special_tokens)
	return _load_SimpleTokenizer

class TestSimpleTokenizer(TestTokenizer):

	@pytest.mark.dependency()
	def test_init(self, load_SimpleTokenizer):
		sentences = [
			'<go> hello ! I am <unk> <eos> <pad>',
			'<go> <unk> why are <unk> here ? <eos>'
		]
		sessions = [
			sentences[:],
			sentences[:] + sentences[:1],
		]
		
		super().base_test_init()
		
		with pytest.raises(ValueError):
			load_SimpleTokenizer('unknown_method')
			
		super().base_test_tokenize_sentences(load_SimpleTokenizer('space'), sentences)
		super().base_test_tokenize_sessions(load_SimpleTokenizer('nltk'), sessions)

	@pytest.mark.dependency(depends=["TestSimpleTokenizer::test_init"])
	def test_convert_tokens_to_sentence(self, load_SimpleTokenizer):
		tokens = ['A', 'beautiful', 'dessert', 'waiting', 'to', 'be', 'shared', 'by', 'two', 'people', '.']
		
		with pytest.raises(RuntimeError):
			toker = load_SimpleTokenizer('space')
			toker.method = 'unknown_method'
			toker.convert_tokens_to_sentence(tokens)
		
		nltk = load_SimpleTokenizer('nltk')
		nltk_res = "A beautiful dessert waiting to be shared by two people."
		assert nltk.convert_tokens_to_sentence(tokens) == nltk_res
		space = load_SimpleTokenizer('space')
		space_res = ' '.join(tokens)
		assert space.convert_tokens_to_sentence(tokens) == space_res

	@pytest.mark.dependency(depends=["TestSimpleTokenizer::test_init"])
	def test_get_setting_hash(self, load_SimpleTokenizer):
		toker1 = load_SimpleTokenizer('space', special_tokens=['<pad>', '<unk>', '<go>', '<eos>'])
		toker2 = load_SimpleTokenizer('space', special_tokens=['<pad>', '<unk>', '<go>', '<eos>'])
		toker3 = load_SimpleTokenizer('nltk', special_tokens=['<pad>', '<unk>', '<go>', '<eos>'])
		toker4 = load_SimpleTokenizer('space', special_tokens=['<pad>', '<unk>', '<go>'])
		toker5 = load_SimpleTokenizer('space', special_tokens=['<pad>', '<eos>', '<unk>', '<go>'])
		assert toker1.get_setting_hash() == toker2.get_setting_hash()
		assert toker1.get_setting_hash() != toker3.get_setting_hash()
		assert toker1.get_setting_hash() != toker4.get_setting_hash()
		assert toker1.get_setting_hash() != toker5.get_setting_hash()


@pytest.fixture
def load_PretrainedTokenizer():
	def _load_PretrainedTokenizer(unk_token='<|endoftext|>'):
		vocab_file = './tests/dataloader/dummy_gpt2vocab/vocab.json'
		merges_file = './tests/dataloader/dummy_gpt2vocab/merges.txt'
		from transformers.tokenization_gpt2 import GPT2Tokenizer
		return PretrainedTokenizer(GPT2Tokenizer(vocab_file, merges_file, unk_token=unk_token))
	return _load_PretrainedTokenizer


class TestPretrainedTokenizer(TestTokenizer):

	@pytest.mark.dependency()
	def test_init(self, load_PretrainedTokenizer):
		sentences = [
			'you know how in some movies . they have a dream sequence , only they dontteii you its a dream ?',
			'Two women waiting at a bench next to a street .'
		]
		sessions = [
			sentences[:],
			sentences[:] + sentences[:1],
		]
		super().base_test_init()
		super().base_test_tokenize_sentences(load_PretrainedTokenizer(), sentences)
		super().base_test_tokenize_sessions(load_PretrainedTokenizer(), sessions)
		
	@pytest.mark.dependency(depends=["TestPretrainedTokenizer::test_init"])
	def test_convert_tokens_to_sentence(self, load_PretrainedTokenizer):
		tokens = ['A', 'Ġbeautiful', 'Ġdessert', 'Ġwaiting', 'Ġto', 'Ġbe', 'Ġshared', 'Ġby', 'Ġtwo', 'Ġpeople', '.']
		toker = load_PretrainedTokenizer()
		res = 'A beautiful dessert waiting to be shared by two people.'
		assert toker.convert_tokens_to_sentence(tokens) == res

	@pytest.mark.dependency(depends=["TestPretrainedTokenizer::test_init"])
	def test_get_setting_hash(self, load_PretrainedTokenizer):
		toker1 = load_PretrainedTokenizer(unk_token='<|endoftext|>')
		toker2 = load_PretrainedTokenizer(unk_token='<|endoftext|>')
		toker3 = load_PretrainedTokenizer(unk_token='<unk>')
		assert toker1.get_setting_hash() == toker2.get_setting_hash()
		assert toker1.get_setting_hash() != toker3.get_setting_hash()
	
	@pytest.mark.dependency(depends=["TestPretrainedTokenizer::test_init"])
	def test_get_tokenizer_class(self, load_PretrainedTokenizer):
		assert load_PretrainedTokenizer().get_tokenizer_class() == 'GPT2Tokenizer'
