import pytest
import requests
import requests_mock
import os
import hashlib
import json

from contk._utils.file_utils import *

@pytest.fixture
def r_mock():
	with requests_mock.Mocker() as m:
		yield m

class TestFileUtils():
	def test_download(self, r_mock):
		r_mock.get('http://coai.cs.tsinghua.edu.cn/', text='coai')

		cache_dir = './tests/_utils/dataset_cache'
		config_dir = './tests/_utils/dummy_coai'
		res_path = get_resource('coai', cache_dir=cache_dir, config_dir=config_dir)
		
		assert(res_path == os.path.join(cache_dir, 'coai'))
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
			assert(meta == {'local_path': res_path})
