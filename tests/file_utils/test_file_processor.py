import pytest
import requests
import requests_mock
import os
import pathlib
import hashlib
import json
import shutil
from checksumdir import dirhash

from cotk.file_utils import get_resource_file_path

@pytest.fixture
def r_mock():
	with requests_mock.Mocker() as m:
		yield m

def check(file1, file2):
	with open(file1, 'r', encoding='utf-8') as f1:
		with open(file2, 'r', encoding='utf-8') as f2:
			assert f1.read() == f2.read()

class TestFileUtils():
	def test_MSCOCO_resource(self):
		cache_dir = str(pathlib.Path('./tests/file_utils/dataset_cache'))
		config_dir = str(pathlib.Path('./tests/file_utils/dummy_coai'))
		data_dir = str(pathlib.Path('./tests/file_utils/data'))
		res_path = get_resource_file_path(str(pathlib.Path('./tests/file_utils/data/mscoco.zip#MSCOCO')))
		assert os.path.isdir(res_path)

		for key in ['train', 'test', 'dev']:
			assert os.path.isfile(os.path.join(res_path, key + '.txt'))
		shutil.rmtree(str(pathlib.Path('./tests/file_utils/data/mscoco.zip_unzip')))

	def test_OpenSubtitles_resource(self):
		cache_dir = str(pathlib.Path('./tests/file_utils/dataset_cache'))
		config_dir = str(pathlib.Path('./tests/file_utils/dummy_coai'))
		data_dir = str(pathlib.Path('./tests/file_utils/data'))
		res_path = get_resource_file_path(str(pathlib.Path('./tests/file_utils/data/opensubtitles.zip#OpenSubtitles')))
		assert os.path.isdir(res_path)

		for key in ['train', 'test', 'dev']:
			assert os.path.isfile(os.path.join(res_path, key + '.txt'))

		shutil.rmtree(str(pathlib.Path('./tests/file_utils/data/opensubtitles.zip_unzip')))

	def test_Ubuntu_resource(self):
		cache_dir = str(pathlib.Path('./tests/file_utils/dataset_cache'))
		config_dir = str(pathlib.Path('./tests/file_utils/dummy_coai'))
		data_dir = str(pathlib.Path('./tests/file_utils/data'))
		res_path = get_resource_file_path(str(pathlib.Path('./tests/file_utils/data/ubuntu_dataset.zip#Ubuntu')))
		assert os.path.isdir(res_path)

		for key in ['train', 'test', 'dev']:
			assert os.path.isfile(os.path.join(res_path, key + '.txt'))

		shutil.rmtree(str(pathlib.Path('./tests/file_utils/data/ubuntu_dataset.zip_unzip')))

	def test_SwitchboardCorpus_resource(self):
		cache_dir = str(pathlib.Path('./tests/file_utils/dataset_cache'))
		config_dir = str(pathlib.Path('./tests/file_utils/dummy_coai'))
		data_dir = str(pathlib.Path('./tests/file_utils/data'))
		res_path = get_resource_file_path(str(pathlib.Path('./tests/file_utils/data/switchboard_corpus.zip#SwitchboardCorpus')))
		assert os.path.isdir(res_path)

		for key in ['train', 'test', 'dev', 'multi_ref']:
			assert os.path.isfile(os.path.join(res_path, key + '.txt'))

		shutil.rmtree(str(pathlib.Path('./tests/file_utils/data/switchboard_corpus.zip_unzip')))

	def test_glove50d_resource(self):
		cache_dir = str(pathlib.Path('./tests/file_utils/dataset_cache'))
		config_dir = str(pathlib.Path('./tests/file_utils/dummy_coai'))
		data_dir = str(pathlib.Path('./tests/file_utils/data'))
		res_path = get_resource_file_path(str(pathlib.Path('./tests/file_utils/data/glove.6B.50d.zip#Glove50d')))

		filenames = os.listdir(res_path)
		assert res_path == str(pathlib.Path('./tests/file_utils/data/glove.6B.50d.zip_unzip/50d'))
		assert sorted(filenames) == sorted(os.listdir(os.path.join(data_dir, 'glove', '50d')))
		for filename in filenames:
			check(os.path.join(res_path, filename), os.path.join(data_dir, 'glove', '50d', filename))

		shutil.rmtree(str(pathlib.Path('./tests/file_utils/data/glove.6B.50d.zip_unzip')))

