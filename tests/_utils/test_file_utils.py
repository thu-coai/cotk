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

class TestFileUtils():
	def test_get_resource(self, r_mock):
		r_mock.get('http://coai.cs.tsinghua.edu.cn/', text='coai')

		cache_dir = './tests/_utils/dataset_cache'
		config_dir = './tests/_utils/dummy_coai'
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

	def test_download_resource(self):
		cache_dir = './tests/_utils/dataset_cache'
		config_dir = './tests/_utils/dummy_coai'
		res_path = get_resource_file_path('resources://MSCOCO#MSCOCO', cache_dir=cache_dir, config_dir=config_dir)

		assert(res_path == os.path.join(cache_dir, '8a33092c80383e264dfbe3ab191b110f9d5a0b644f387ae910e0b441228bd8ff_unzip/mscoco'))
		assert(os.path.exists(res_path))

		meta_path = os.path.join(cache_dir, '8a33092c80383e264dfbe3ab191b110f9d5a0b644f387ae910e0b441228bd8ff.json')
		assert(os.path.exists(meta_path))
		with open(meta_path, 'r') as meta_file:
			meta = json.load(meta_file)
			assert(meta['local_path'] == res_path)

