import pytest
import requests
import requests_mock
import os
import hashlib
import json
import shutil
from checksumdir import dirhash

from cotk._utils.file_utils import get_resource_file_path, import_local_resources

@pytest.fixture
def r_mock():
	with requests_mock.Mocker() as m:
		yield m

class TestFileUtils():
	def test_get_resource(self, r_mock):
		r_mock.get('http://coai.cs.tsinghua.edu.cn/', text='coai')

		cache_dir = './tests/_utils/dataset_cache'
		config_dir = './tests/_utils/dummy_coai'

		with pytest.raises(FileNotFoundError) as excinfo:
			get_resource_file_path('resources://coai', cache_dir=cache_dir, config_dir='wrongpath')
		assert("not found" in str(excinfo.value))

		with pytest.raises(ValueError) as excinfo:
			get_resource_file_path('resources://coai#wrongtype', cache_dir=cache_dir, config_dir=config_dir)
		assert("differs with res_type" in str(excinfo.value))

		with pytest.raises(ValueError) as excinfo:
			get_resource_file_path('resources://coai@wronglink', cache_dir=cache_dir, config_dir=config_dir)
		assert("source wronglink wrong" in str(excinfo.value))

		res_path = get_resource_file_path('resources://coai', cache_dir=cache_dir, config_dir=config_dir)
		
		assert(res_path == os.path.join(cache_dir, '146ce545f2ed0a8767aadae8f2921f7951df817b39b8f7d0db48bce87e3eaf69'))
		assert(os.path.exists(res_path))

		hash_sha256 = hashlib.sha256()
		with open(res_path, "rb") as fin:
			for chunk in iter(lambda: fin.read(4096), b""):
				hash_sha256.update(chunk)
		assert(hash_sha256.hexdigest() == "146ce545f2ed0a8767aadae8f2921f7951df817b39b8f7d0db48bce87e3eaf69")

		meta_path = res_path + '.json'
		assert(os.path.exists(meta_path))
		with open(meta_path, 'r') as meta_file:
			meta = json.load(meta_file)
			assert(meta['local_path'] == res_path)
		
		shutil.rmtree(cache_dir)

	def test_download_resource(self):
		cache_dir = './tests/_utils/dataset_cache'
		config_dir = './tests/_utils/dummy_coai'
		res_path = get_resource_file_path('resources://test@amazon', cache_dir=cache_dir, config_dir=config_dir)
		res_path = get_resource_file_path('resources://test', cache_dir=cache_dir, config_dir=config_dir)

		assert(res_path == os.path.join(cache_dir, '9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08'))
		assert(os.path.exists(res_path))

		meta_path = os.path.join(cache_dir, '9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08.json')
		assert(os.path.exists(meta_path))
		with open(meta_path, 'r') as meta_file:
			meta = json.load(meta_file)
			assert(meta['local_path'] == res_path)
		
		shutil.rmtree(cache_dir)

	def test_download_data(self):
		cache_dir = './tests/_utils/dataset_cache'
		config_dir = './tests/_utils/dummy_coai'
		res_path = get_resource_file_path('https://cotk-data.s3-ap-northeast-1.amazonaws.com/test.zip', cache_dir=cache_dir)
		res_path = get_resource_file_path('https://cotk-data.s3-ap-northeast-1.amazonaws.com/test.zip', cache_dir=cache_dir)

		assert(res_path == os.path.join(cache_dir, 'f1043836933af4b8b28973d259c0c77f5049de2dff8d0d1f305c65f3c497b3b1'))
		assert(os.path.exists(res_path))

		meta_path = os.path.join(cache_dir, 'f1043836933af4b8b28973d259c0c77f5049de2dff8d0d1f305c65f3c497b3b1.json')
		assert(os.path.exists(meta_path))
		with open(meta_path, 'r') as meta_file:
			meta = json.load(meta_file)
			assert(meta['local_path'] == res_path)
		
		shutil.rmtree(cache_dir)

	def test_import_local_resource(self):
		cache_dir = './tests/_utils/dataset_cache'
		config_dir = './tests/_utils/dummy_coai'
		data_dir = './tests/_utils/data'

		with pytest.raises(ValueError) as excinfo:
			import_local_resources('test', local_path=os.path.join(data_dir, 'test.zip'), cache_dir=cache_dir, config_dir=config_dir)
		assert("file_id must startswith" in str(excinfo.value))

		with pytest.raises(ValueError) as excinfo:
			import_local_resources('resources://test', local_path=os.path.join(data_dir, 'mscoco.zip'), cache_dir=cache_dir, config_dir=config_dir)
		assert("bad hashtag" in str(excinfo.value))

		res_path = import_local_resources('resources://test', local_path=os.path.join(data_dir, 'test.zip'), cache_dir=cache_dir, config_dir=config_dir)

		with pytest.raises(ValueError) as excinfo:
			import_local_resources('resources://test', local_path=os.path.join(data_dir, 'test.zip'), cache_dir=cache_dir, config_dir=config_dir, ignore_exist_error=True)
			import_local_resources('resources://test', local_path=os.path.join(data_dir, 'test.zip'), cache_dir=cache_dir, config_dir=config_dir, ignore_exist_error=False)		
		assert("resources existed. If you want to delete the existing resources." in str(excinfo.value))

		assert(res_path == os.path.join(cache_dir, '9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08'))
		assert(os.path.exists(res_path))

		meta_path = os.path.join(cache_dir, '9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08.json')
		assert(os.path.exists(meta_path))
		with open(meta_path, 'r') as meta_file:
			meta = json.load(meta_file)
			assert(meta['local_path'] == res_path)
		
		shutil.rmtree(cache_dir)
