import os
import pytest
from cotk.scripts.main import dispatch
import re
import json
import subprocess
from subprocess import PIPE
import shutil
import argparse

class TestScripts():
	i = 0

	def setup(self):
		try:
			shutil.rmtree("cotk-test-CVAE")
		except FileNotFoundError:
			pass
		except PermissionError:
			os.rename("cotk-test-CVAE", "cotk-test-CVAE" + str(TestScripts.i))
			TestScripts.i += 1

	@pytest.mark.parametrize('url, error, match', \
		[("http://wrong_url", ValueError, "can't match any pattern"),\
		('user/repo/commit/wrong_commit', RuntimeError, "fatal"),\
		('http://github.com/thu-coai/cotk-test-CVAE/no_result_file/', FileNotFoundError, r"Config file .* is not found."),\
		('https://github.com/thu-coai/cotk-test-CVAE/tree/invalid_json', json.JSONDecodeError, ""),\
		])
	def test_download_error(self, url, error, match):
		with pytest.raises(error, match=match):
			dispatch('download', [url])

	def test_download(self):
		# with pytest.raises(FileNotFoundError) as excinfo:
		# 	report.dispatch('download', \
		# 					['--zip_url', 'https://github.com/thu-coai/cotk-test-CVAE/archive/no_output.zip'])
		# assert "New result file not found." == str(excinfo.value)
		dispatch('download', ['https://github.com/thu-coai/cotk-test-CVAE/tree/run_and_test'])

	def test_config(self):
		dispatch('config', ["set", 'token', "123"])

	def test_import_local_resources(self):
		shutil.copyfile('./tests/_utils/dummy_coai/test.json', './cotk/resource_config/test.json')

		dispatch('import', ['resources://test', './tests/_utils/data/test.zip'])

		os.remove('./cotk/resource_config/test.json')
