import pytest
import requests
import requests_mock
import os
import hashlib
import json
from checksumdir import dirhash

from cotk._utils.file_utils import get_resource_file_path

@pytest.fixture
def r_mock():
	with requests_mock.Mocker() as m:
		yield m

def check(file1, file2):
	with open(file1, 'r') as f1:
		with open(file2, 'r') as f2:
			assert(f1.read() == f2.read())

class TestFileUtils():
	def test_MSCOCO_resource(self):
		cache_dir = './tests/_utils/dataset_cache'
		config_dir = './tests/_utils/dummy_coai'
		data_dir = './tests/_utils/data'
		res_path = get_resource_file_path('./tests/_utils/data/mscoco.zip#MSCOCO', cache_dir=cache_dir, config_dir=config_dir)

		filenames = os.listdir(res_path)
		assert(sorted(filenames) == sorted(os.listdir(os.path.join(data_dir, 'mscoco'))))
		for filename in filenames:
			check(os.path.join(res_path, filename), os.path.join(data_dir, 'mscoco', filename))

	def test_OpenSubtitles_resource(self):
		cache_dir = './tests/_utils/dataset_cache'
		config_dir = './tests/_utils/dummy_coai'
		data_dir = './tests/_utils/data'
		res_path = get_resource_file_path('./tests/_utils/data/opensubtitles.zip#OpenSubtitles', cache_dir=cache_dir, config_dir=config_dir)

		filenames = os.listdir(res_path)
		assert(sorted(filenames) == sorted(os.listdir(os.path.join(data_dir, 'opensubtitles'))))
		for filename in filenames:
			check(os.path.join(res_path, filename), os.path.join(data_dir, 'opensubtitles', filename))

	def test_Ubuntu_resource(self):
		cache_dir = './tests/_utils/dataset_cache'
		config_dir = './tests/_utils/dummy_coai'
		data_dir = './tests/_utils/data'
		res_path = get_resource_file_path('./tests/_utils/data/ubuntu_dataset.zip#Ubuntu', cache_dir=cache_dir, config_dir=config_dir)
		filenames = os.listdir(res_path)
		assert(sorted(filenames) == sorted(os.listdir(os.path.join(data_dir, 'ubuntu_dataset'))))
		for filename in filenames:
			check(os.path.join(res_path, filename), os.path.join(data_dir, 'ubuntu_dataset', filename))

	def test_SwitchboardCorpus_resource(self):
		cache_dir = './tests/_utils/dataset_cache'
		config_dir = './tests/_utils/dummy_coai'
		data_dir = './tests/_utils/data'
		res_path = get_resource_file_path('./tests/_utils/data/switchboard_corpus.zip#SwitchboardCorpus', cache_dir=cache_dir, config_dir=config_dir)
		filenames = os.listdir(res_path)
		assert(sorted(filenames) == sorted(os.listdir(os.path.join(data_dir, 'switchboard_corpus'))))
		for filename in filenames:
			check(os.path.join(res_path, filename), os.path.join(data_dir, 'switchboard_corpus', filename))

	def test_glove50d_resource(self):
		cache_dir = './tests/_utils/dataset_cache'
		config_dir = './tests/_utils/dummy_coai'
		data_dir = './tests/_utils/data'
		res_path = get_resource_file_path('./tests/_utils/data/glove.6B.50d.zip#Glove50d', cache_dir=cache_dir, config_dir=config_dir)

		filenames = os.listdir(res_path)
		print(res_path)
		print(filenames)
		assert(sorted(filenames) == sorted(os.listdir(os.path.join(data_dir, 'glove', '50d'))))
		for filename in filenames:
			check(os.path.join(res_path, filename), os.path.join(data_dir, 'glove', '50d', filename))

