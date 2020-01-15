import itertools
import copy
import hashlib
from unittest import mock

import pytest

import cotk
from cotk.dataloader.dataloader import LanguageProcessingBase, Label, Sentence, Session, DataField


@pytest.fixture
def get_fake_dataset():
	ext_vocab = ["<pad>", "<unk>", "<go>", "<eos>"]
	vocabs = ['to', 'in', 'designing', 'building', 'on', 'is', 'provides', 'cotk', 'you', 'of', 'model', 'the', 'suites',
		'make', 'models', 'dataset', 'language', 'standard', 'source', 'lightweight', 'it', 'open', 'use', 'for',
		'framework', 'and', 'evaluation', 'an', 'general', 'easy', 'focus', 'your', 'generation', 'we', 'domain']
	all_vocabs = ext_vocab + vocabs
	word2index = dict(zip(all_vocabs, range(len(all_vocabs))))

	sentences = ["cotk is an open source lightweight framework for model building and evaluation",
				 "we provides standard dataset and evaluation suites in the domain of general language generation",
				 "it is easy to use and make you focus on designing your models"]
	sentences = [[word2index[vocab] for vocab in sentence.split()] for sentence in sentences]
	sessions = []
	for session_turn_length in range(1, len(sentences) + 1):
		for indices in itertools.combinations(range(len(sentences)), session_turn_length):
			sessions.append([sentences[i] for i in indices])
	labels = list(range(5))

	def get_batch(list_, batch=10):
		rv = []
		while batch > len(list_):
			rv += list_
			batch -= len(list_)
		rv += list_[:batch]
		return rv

	data = {  # 'sentence', 'session', 'label' have the same length.
		'sentence': get_batch(sentences),
		'session': get_batch(sessions),
		'label': get_batch(labels)
	}

	def generate_dataset(fields, data):
		# example fields = [['sent', 'sentence'], ['label', 'label']]
		return {data_key: data[field.lower()] for data_key, field in fields}

	key_name = 'train test valid'.split()
	fields = {
		'train': [['sent', 'Sentence'], ['session', 'Session'], ['label', 'Label']],
		'valid': [['sent', 'Sentence'], ['session', 'Session'], ['label', 'Label']],
		'test': [['sent', 'Sentence'], ['session', 'Session']]
	}
	fake_dataset = {}
	for key in key_name:
		fake_dataset[key] = generate_dataset(fields[key], data)

	return all_vocabs, word2index, key_name, fields, fake_dataset


class FakeDataloader(LanguageProcessingBase):
	"""
	A fake dataloader for testing class `DataloaderHash`.
	"""
	def __init__(self,
				filepath,
				fields,
				min_vocab_times,
				max_sent_length,
				max_turn_length,
				invalid_vocab_times,
				ext_vocab=None,
				key_name=None):
		self.filepath = filepath
		self.data_fields = fields
		self.min_vocab_times = min_vocab_times
		self.max_sent_length = max_sent_length
		self.max_turn_length = max_turn_length
		self.invalid_vocab_times = invalid_vocab_times
		super().__init__(ext_vocab, key_name, tokenizer='space')

	def get_batch(self, key, indexes):
		pass

	def _load_data(self):
		return super()._general_load_data(self.filepath,
										  self.data_fields,
										  self.min_vocab_times,
										  self.max_sent_length,
										  self.max_turn_length,
										  self.invalid_vocab_times)


class TestHash:
	def _hash(self, ext_vocab_ids, unk_id, valid_vocabs, data_fields, dataset):
		ignore_ids = [i for i in ext_vocab_ids if i != unk_id]
		with mock.patch('cotk.dataloader.dataloader.DataloaderHash', side_effect=cotk.dataloader.dataloader.DataloaderHash):
			hash_obj = cotk.dataloader.dataloader.DataloaderHash(ignore_tokens=ignore_ids, unk_id=unk_id)
			hash_value = hash_obj.hash_datasets(dataset, data_fields, valid_vocabs)
			assert cotk.dataloader.dataloader.DataloaderHash.called
			return hash_value

	@pytest.mark.dependency()
	def test_hash(self, get_fake_dataset):
		all_vocabs, word2index, key_name, data_fields, fake_dataset = get_fake_dataset
		ext_ids = list(range(4))
		unk_id = 1
		ignore_ids = [i for i in ext_ids if i != unk_id]
		hash_value = self._hash(ext_ids, unk_id, all_vocabs, data_fields, fake_dataset)

		hash_obj = hashlib.sha256()
		for key in sorted(key_name):
			bytes_ = cotk.dataloader.dataloader.DataloaderHash(ignore_ids, unk_id)._hash_dataset(fake_dataset[key], data_fields[key], all_vocabs)
			hash_obj.update(bytes_)
		assert hash_obj.hexdigest() == hash_value  # test _hash_dataset
		# test whether `hash_datasets` and `self._hash` return the same value
		assert hash_value == cotk.dataloader.dataloader.DataloaderHash(ignore_ids, unk_id).hash_datasets(fake_dataset, data_fields, all_vocabs)

	@staticmethod
	def insert_token(dataset, data_fields, tokens):
		tokens = list(tokens)
		dataset2 = copy.deepcopy(dataset)  # add ext_vocab
		for key in dataset2:
			new_data = {}
			for data_key, field in data_fields[key]:
				data = dataset2[key][data_key]
				field = DataField.get_field(field)
				if isinstance(field, Sentence):
					data = [tokens + sentence + tokens * 2 for sentence in data]
				elif isinstance(field, Session):
					data = [[tokens + sentence + tokens * 3 for sentence in session] for session in data]
				new_data[data_key] = data
			dataset2[key] = new_data
		return dataset2

	@pytest.mark.dependency()
	def test_ignore_ext_vocab(self, get_fake_dataset):
		"""
		Test whether DataloaderHash ignore ext_vocab.
		"""
		all_vocabs, word2index, key_name, data_fields, fake_dataset = get_fake_dataset
		ext_ids = list(range(4))
		pad_id, unk_id, go_id, eos_id = ext_ids

		hash_value = self._hash(ext_ids, unk_id, all_vocabs, data_fields, fake_dataset)
		fake_dataset2 = self.insert_token(fake_dataset, data_fields, [pad_id, eos_id, go_id])
		assert hash_value == self._hash(ext_ids, unk_id, all_vocabs, data_fields, fake_dataset2)

	@pytest.mark.dependency()
	def test_valid_vocab(self, get_fake_dataset):
		"""
		Test whether hash values are same when valid vocabs are same, but all vocabs are different.
		"""
		valid_vocabs, word2index, key_name, data_fields, fake_dataset = get_fake_dataset
		ext_ids = list(range(4))
		unk_id = 1

		valid_vocab_num = len(valid_vocabs)
		invalid_ids1 = list(range(valid_vocab_num, valid_vocab_num + 10))
		invalid_ids2 = list(range(valid_vocab_num + 10, valid_vocab_num + 20))

		# fake_dataset1 and fake_dataset2 has the same valid_vocabs but different vocabs
		fake_dataset1 = self.insert_token(fake_dataset, data_fields, invalid_ids1)
		fake_dataset2 = self.insert_token(fake_dataset, data_fields, invalid_ids2)

		hash_value1 = self._hash(ext_ids, unk_id, valid_vocabs, data_fields, fake_dataset1)
		assert hash_value1 == \
			self._hash(ext_ids, unk_id, valid_vocabs, data_fields, fake_dataset2)
		assert hash_value1 != self._hash(ext_ids, unk_id, valid_vocabs[:-1], data_fields, fake_dataset2)
