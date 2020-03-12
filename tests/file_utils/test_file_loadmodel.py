import pytest
import requests
import requests_mock
import shutil
import os

from cotk.file_utils.file_utils import _get_file_sha256
from cotk.file_utils import load_file_from_url

@pytest.fixture
def r_mock():
	with requests_mock.Mocker() as m:
		yield m

class TestFileUtils():
	def test_MSCOCO_resource(self):
		cache_dir = './tests/file_utils/dataset_cache'
		data_dir = './tests/file_utils/data'
		res_path = load_file_from_url('https://cotk-data.s3-ap-northeast-1.amazonaws.com/test.zip', cache_dir=cache_dir)

		res_path2 = load_file_from_url('https://cotk-data.s3-ap-northeast-1.amazonaws.com/test.zip', cache_dir=cache_dir)
		assert res_path == res_path2

		assert res_path == os.path.join(cache_dir, 'files', 'test.zip')
		assert _get_file_sha256(res_path) == _get_file_sha256(os.path.join(data_dir, 'test.zip'))

		shutil.rmtree(cache_dir)
