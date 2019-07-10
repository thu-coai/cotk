import os
import pytest
from cotk.scripts import entry
import re
import json
import subprocess
from subprocess import PIPE

class TestScripts():

	@pytest.mark.parametrize('url, error, match', \
		[("http://wrong_url", ValueError, "can't match any pattern"),\
		('user/repo/commit/wrong_commit', RuntimeError, "It should be public."),\
		('http://github.com/thu-coai/cotk-test-CVAE/no_result_file/', FileNotFoundError, r"Config file .* is not found."),\
		('https://github.com/thu-coai/cotk-test-CVAE/tree/invalid_json', json.JSONDecodeError, ""),\
		('thu-coai/cotk-test-CVAE/tree/keys_undefined', RuntimeError, "Undefined keys"),\
		])
	def test_download_error(self, url, error, match):
		with pytest.raises(error, match=match):
			entry.dispatch('download', [url])

	def test_download(self):
		# with pytest.raises(FileNotFoundError) as excinfo:
		# 	report.dispatch('download', \
		# 					['--zip_url', 'https://github.com/thu-coai/cotk-test-CVAE/archive/no_output.zip'])
		# assert "New result file not found." == str(excinfo.value)
		entry.dispatch('download', ['https://github.com/thu-coai/cotk-test-CVAE/tree/run_and_test'])
