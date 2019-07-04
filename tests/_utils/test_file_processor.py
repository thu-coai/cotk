import pytest
import requests
import requests_mock
import os
import hashlib
import json
import shutil
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
		res_path = get_resource_file_path('./tests/_utils/data/mscoco.zip#MSCOCO')

		filenames = os.listdir(res_path)
		assert(res_path == './tests/_utils/data/mscoco.zip_unzip/mscoco')
		assert(sorted(filenames) == sorted(os.listdir(os.path.join(data_dir, 'mscoco'))))
		for filename in filenames:
			check(os.path.join(res_path, filename), os.path.join(data_dir, 'mscoco', filename))

		shutil.rmtree('./tests/_utils/data/mscoco.zip_unzip')

	def test_OpenSubtitles_resource(self):
		cache_dir = './tests/_utils/dataset_cache'
		config_dir = './tests/_utils/dummy_coai'
		data_dir = './tests/_utils/data'
		res_path = get_resource_file_path('./tests/_utils/data/opensubtitles.zip#OpenSubtitles')

		filenames = os.listdir(res_path)
		assert(res_path == './tests/_utils/data/opensubtitles.zip_unzip/opensubtitles')
		assert(sorted(filenames) == sorted(os.listdir(os.path.join(data_dir, 'opensubtitles'))))
		for filename in filenames:
			check(os.path.join(res_path, filename), os.path.join(data_dir, 'opensubtitles', filename))

		shutil.rmtree('./tests/_utils/data/opensubtitles.zip_unzip')

	def test_Ubuntu_resource(self):
		cache_dir = './tests/_utils/dataset_cache'
		config_dir = './tests/_utils/dummy_coai'
		data_dir = './tests/_utils/data'
		res_path = get_resource_file_path('./tests/_utils/data/ubuntu_dataset.zip#Ubuntu')

		filenames = os.listdir(res_path)
		assert(res_path == './tests/_utils/data/ubuntu_dataset.zip_unzip/ubuntu_dataset')
		assert(sorted(filenames) == sorted(os.listdir(os.path.join(data_dir, 'ubuntu_dataset'))))
		for filename in filenames:
			check(os.path.join(res_path, filename), os.path.join(data_dir, 'ubuntu_dataset', filename))

		shutil.rmtree('./tests/_utils/data/ubuntu_dataset.zip_unzip')

	def test_SwitchboardCorpus_resource(self):
		cache_dir = './tests/_utils/dataset_cache'
		config_dir = './tests/_utils/dummy_coai'
		data_dir = './tests/_utils/data'
		res_path = get_resource_file_path('./tests/_utils/data/switchboard_corpus.zip#SwitchboardCorpus')

		filenames = os.listdir(res_path)
		assert(res_path == './tests/_utils/data/switchboard_corpus.zip_unzip/switchboard_corpus')
		assert(sorted(filenames) == sorted(os.listdir(os.path.join(data_dir, 'switchboard_corpus'))))
		for filename in filenames:
			check(os.path.join(res_path, filename), os.path.join(data_dir, 'switchboard_corpus', filename))

		shutil.rmtree('./tests/_utils/data/switchboard_corpus.zip_unzip')

	def test_glove50d_resource(self):
		cache_dir = './tests/_utils/dataset_cache'
		config_dir = './tests/_utils/dummy_coai'
		data_dir = './tests/_utils/data'
		res_path = get_resource_file_path('./tests/_utils/data/glove.6B.50d.zip#Glove50d')

		filenames = os.listdir(res_path)
		assert(res_path == './tests/_utils/data/glove.6B.50d.zip_unzip/50d')
		assert(sorted(filenames) == sorted(os.listdir(os.path.join(data_dir, 'glove', '50d'))))
		for filename in filenames:
			check(os.path.join(res_path, filename), os.path.join(data_dir, 'glove', '50d', filename))

		shutil.rmtree('./tests/_utils/data/glove.6B.50d.zip_unzip')

