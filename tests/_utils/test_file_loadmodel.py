import pytest
import requests
import requests_mock
import shutil
import os

from cotk._utils.file_utils import load_model_from_url, _get_file_sha256

@pytest.fixture
def r_mock():
	with requests_mock.Mocker() as m:
		yield m

class TestFileUtils():
	def test_MSCOCO_resource(self):
		cache_dir = './tests/_utils/dataset_cache'
		data_dir = './tests/_utils/data'
		res_path = load_model_from_url('https://cotk-data.s3-ap-northeast-1.amazonaws.com/test.zip', cache_dir=cache_dir)

		with pytest.raises(ValueError) as excinfo:
			load_model_from_url('https://cotk-data.s3-ap-northeast-1.amazonaws.com/test.zip', cache_dir=cache_dir)
		assert("model existed. If you want to delete the existing model." in str(excinfo.value))

		assert(res_path == os.path.join(cache_dir, 'models', 'test.zip'))
		assert(_get_file_sha256(res_path) == _get_file_sha256(os.path.join(data_dir, 'test.zip')))

		shutil.rmtree(cache_dir)
